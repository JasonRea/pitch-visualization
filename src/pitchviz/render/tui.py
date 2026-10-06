"""Interactive, colorful terminal UI for building pitch visualization renders.

Wraps the same data/filter/render pipeline the flag-based CLI (`render/run.py`)
uses, but lets the user search for a pitcher, browse their outings, and choose
a full outing / a specific at-bat / specific pitches (pitch tunneling) instead
of typing dates and flags by hand.
"""

import sys
from datetime import date as _date

import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pitchviz.config import PITCH_NAMES
from pitchviz.data.fetch import search_pitchers, get_pitcher_outings, get_player_names, pitch_data, live_pitch_data
from pitchviz.data.sequence import at_bat_sequences
from pitchviz.data.filters import (
    pitches_filter,
    pitches_filter_by_at_bat,
    pitches_filter_by_pitch_numbers,
)
from pitchviz.render.builder import VizualizationBuilder

console = Console()

# Sentinel returned by any step to mean "go back one step" instead of a value.
_BACK = object()

QUALITY_CHOICES = [
    Choice(title="Low (fast, good for previewing)", value="low_quality"),
    Choice(title="Medium", value="medium_quality"),
    Choice(title="High", value="high_quality"),
    Choice(title="4K (slow)", value="fourk_quality"),
]

STYLE = questionary.Style([
    ("qmark", "fg:#67E18D bold"),
    ("question", "bold"),
    ("answer", "fg:#FF007D bold"),
    ("pointer", "fg:#67E18D bold"),
    ("highlighted", "fg:#67E18D bold"),
    ("selected", "fg:#1BB999"),
])


_MIN_STATCAST_YEAR = 2015


def _select_year() -> int:
    """First step — nothing to go back to, so no back option here."""
    current_year = _date.today().year

    def _validate(text: str) -> bool | str:
        if not text.isdigit() or len(text) != 4:
            return "Enter a 4-digit year."
        year = int(text)
        if not (_MIN_STATCAST_YEAR <= year <= current_year):
            return f"Enter a year between {_MIN_STATCAST_YEAR} and {current_year}."
        return True

    answer = questionary.text(
        "Which season?", default=str(current_year), validate=_validate, style=STYLE,
    ).ask()
    if answer is None:
        sys.exit(0)
    return int(answer)


def _select_pitcher(season: int):
    """Returns the confirmed pitcher's match dict, or _BACK."""
    while True:
        query = questionary.text(
            "Search for a pitcher (leave blank to go back):", style=STYLE,
        ).ask()
        if query is None:
            sys.exit(0)
        if query == "":
            return _BACK
        if len(query) < 2:
            console.print("[yellow]Type at least 2 characters.[/yellow]")
            continue

        try:
            matches = search_pitchers(query, season=season)
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            continue

        if not matches:
            console.print(f"[yellow]No {season} pitchers found matching '{query}'.[/yellow]")
            continue

        if len(matches) == 1:
            match = matches[0]
        else:
            choice = questionary.select(
                "Select a pitcher:",
                choices=[
                    Choice(title="← Back", value=_BACK),
                    *[
                        Choice(
                            title=f"{m['full_name']} ({m['team']})" if m["team"] else m["full_name"],
                            value=m["full_name"],
                        )
                        for m in matches
                    ],
                ],
                style=STYLE,
            ).ask()
            if choice is None:
                sys.exit(0)
            if choice is _BACK:
                continue  # re-prompt the search query
            match = next(m for m in matches if m["full_name"] == choice)

        label = f"{match['full_name']} ({match['team']})" if match["team"] else match["full_name"]
        confirmed = questionary.confirm(f"Use {label}?", default=True, style=STYLE).ask()
        if confirmed is None:
            sys.exit(0)
        if confirmed:
            return match
        # not confirmed — loop back to search


def _select_outing(pitcher_name: str, season: int):
    try:
        outings = get_pitcher_outings(pitcher_name, season)
    except Exception as e:
        console.print(f"[red]Could not fetch outings for {pitcher_name}: {e}[/red]")
        sys.exit(1)

    if not outings:
        console.print(f"[yellow]No {season} outings found for {pitcher_name}.[/yellow]")
        sys.exit(1)

    def _title(o: dict) -> str:
        base = f"{o['date']}  vs {o['opponent']} ({o['home_away']})"
        if not o["final"]:
            base += "  (Live)"
        return f"★ {base} (Postseason)" if o["postseason"] else base

    choice = questionary.select(
        f"Select an outing ({pitcher_name}, {season}):",
        choices=[
            Choice(title="← Back", value=_BACK),
            *[Choice(title=_title(o), value=o) for o in reversed(outings)],
        ],
        style=STYLE,
    ).ask()
    if choice is None:
        sys.exit(0)
    return choice


def _format_pitch_choice(p: dict) -> str:
    name = PITCH_NAMES.get(p["pitch_type"], p["pitch_type"])
    velo = f"{p['release_speed']:.1f} mph" if p["release_speed"] is not None else "velo n/a"
    return f"Pitch {p['pitch_number']}: {name} - {velo} - {p['description']}"


def _select_scope(df):
    """Returns a (filter_fn, description, sequential) tuple, or _BACK.

    Internally a small 3-step loop (mode -> at-bat -> pitches) so backing out
    of the pitch checkbox returns to the at-bat list, and backing out of the
    at-bat list returns to the mode choice, without leaving this function —
    only backing out of the mode choice itself returns _BACK to the caller.
    """
    sub_steps = ["mode", "at_bat", "pitches"]
    i = 0
    mode = None
    ab_choice = None
    at_bat = None

    while True:
        step = sub_steps[i]

        if step == "mode":
            mode = questionary.select(
                "What would you like to render?",
                choices=[
                    Choice(title="← Back", value=_BACK),
                    Choice(title="Full outing - every pitch thrown", value="full"),
                    Choice(title="A specific at-bat", value="at_bat"),
                    Choice(title="Specific pitches from a specific at-bat", value="tunnel"),
                ],
                style=STYLE,
            ).ask()
            if mode is None:
                sys.exit(0)
            if mode is _BACK:
                return _BACK
            if mode == "full":
                return pitches_filter, "Full outing", False
            i += 1
            continue

        if step == "at_bat":
            at_bats = at_bat_sequences(df)
            if not at_bats:
                console.print("[yellow]No at-bats found in this outing.[/yellow]")
                i -= 1
                continue

            try:
                batter_names = get_player_names([ab["batter"] for ab in at_bats])
            except Exception:
                batter_names = {}

            ab_choice = questionary.select(
                "Select an at-bat:",
                choices=[
                    Choice(title="← Back", value=_BACK),
                    *[
                        Choice(
                            title=(
                                f"AB #{ab['at_bat_number']} - Inning {ab['inning']} ({ab['inning_topbot']}) "
                                f"vs {batter_names.get(ab['batter'], 'Unknown batter')} ({ab['stand']}) "
                                f"- {ab['final_outcome'] or 'in progress'}"
                            ),
                            value=ab["at_bat_number"],
                        )
                        for ab in at_bats
                    ],
                ],
                style=STYLE,
            ).ask()
            if ab_choice is None:
                sys.exit(0)
            if ab_choice is _BACK:
                i -= 1
                continue

            at_bat = next(ab for ab in at_bats if ab["at_bat_number"] == ab_choice)
            batter_label = batter_names.get(at_bat["batter"], "Unknown batter")

            if mode == "at_bat":
                label = f"vs {batter_label} — Inning {at_bat['inning']}"
                return pitches_filter_by_at_bat(ab_choice, label=label), f"At-Bat #{ab_choice}", True

            i += 1  # tunnel mode continues on to pitch selection
            continue

        if step == "pitches":
            pitch_choices = questionary.checkbox(
                "Select 2+ pitches to overlay (tunnel view), or none to go back:",
                choices=[
                    Choice(title=_format_pitch_choice(p), value=p["pitch_number"])
                    for p in at_bat["pitches"]
                ],
                style=STYLE,
            ).ask()
            if pitch_choices is None:
                sys.exit(0)
            if not pitch_choices:
                i -= 1
                continue
            return pitches_filter_by_pitch_numbers(ab_choice, pitch_choices), f"Tunnel view (AB #{ab_choice})", True


def _select_camera():
    camera = questionary.select(
        "Camera angle:",
        choices=[
            Choice(title="← Back", value=_BACK),
            Choice(title="Catcher's view - looking from home plate toward the mound", value="catcher"),
            Choice(title="Pitcher's mound - looking from the mound toward home plate", value="mound"),
        ],
        style=STYLE,
    ).ask()
    if camera is None:
        sys.exit(0)
    return camera


def _select_quality():
    quality = questionary.select(
        "Render quality:",
        choices=[Choice(title="← Back", value=_BACK), *QUALITY_CHOICES],
        style=STYLE,
    ).ask()
    if quality is None:
        sys.exit(0)
    return quality


def run_tui() -> None:
    console.print(Panel.fit("[bold]Pitch Viz[/bold] — interactive renderer", border_style="#67E18D"))

    results: dict = {}
    steps = ["year", "pitcher", "outing", "scope", "camera", "quality"]
    i = 0

    while i < len(steps):
        step = steps[i]

        if step == "year":
            results["year"] = _select_year()
            i += 1
            continue

        if step == "pitcher":
            pitcher = _select_pitcher(results["year"])
            if pitcher is _BACK:
                i -= 1
                continue
            results["pitcher"] = pitcher
            i += 1
            continue

        if step == "outing":
            outing = _select_outing(results["pitcher"]["full_name"], results["year"])
            if outing is _BACK:
                i -= 1
                continue

            pitcher_name = results["pitcher"]["full_name"]
            date = outing["date"]
            status_verb = "Fetching live" if not outing["final"] else "Fetching"
            with console.status(f"[bold]{status_verb} {pitcher_name}'s outing on {date}...[/bold]"):
                try:
                    if outing["final"]:
                        df = pitch_data(start_dt=date, pitcher=pitcher_name)
                    else:
                        df = live_pitch_data(game_pk=outing["game_pk"], pitcher=pitcher_name)
                except RuntimeError as e:
                    console.print(f"[red]{e}[/red]")
                    sys.exit(1)

            if df.empty:
                console.print(f"[yellow]No pitch data found for {pitcher_name} on {date}.[/yellow]")
                continue  # re-prompt outing selection

            results["date"] = date
            results["df"] = df
            i += 1
            continue

        if step == "scope":
            scope = _select_scope(results["df"])
            if scope is _BACK:
                i -= 1
                continue
            results["filt"], results["description"], results["sequential"] = scope
            i += 1
            continue

        if step == "camera":
            camera = _select_camera()
            if camera is _BACK:
                i -= 1
                continue
            results["camera"] = camera
            i += 1
            continue

        if step == "quality":
            quality = _select_quality()
            if quality is _BACK:
                i -= 1
                continue
            results["quality"] = quality
            i += 1
            continue

    pitcher_name = results["pitcher"]["full_name"]
    date = results["date"]
    description = results["description"]
    camera = results["camera"]
    quality = results["quality"]

    summary = Table.grid(padding=(0, 2))
    summary.add_row("[bold]Pitcher:[/bold]", pitcher_name)
    summary.add_row("[bold]Date:[/bold]", date)
    summary.add_row("[bold]Scope:[/bold]", description)
    summary.add_row("[bold]Camera:[/bold]", "Pitcher's mound" if camera == "mound" else "Catcher's view")
    summary.add_row("[bold]Quality:[/bold]", quality)
    console.print(Panel(summary, title="Render Summary", border_style="#1BB999"))

    builder = VizualizationBuilder()
    builder.load_pitches_from_df(results["df"], results["filt"])
    if builder._axes is None:
        console.print("[red]No pitches to render for this selection.[/red]")
        sys.exit(1)
    scene_class = builder.buildm_pitches(sequential=results["sequential"], camera=camera)
    camera_label = "Mound POV" if camera == "mound" else "Catcher POV"
    filename = f"{pitcher_name} {date} {description} ({camera_label})"

    with console.status("[bold green]Rendering...[/bold green]", spinner="dots"):
        VizualizationBuilder.render(scene_class, quality=quality, filename=filename)

    console.print(Panel(f"[bold green]Done![/bold green] Rendered as '{filename}'", border_style="green"))


if __name__ == "__main__":
    run_tui()
