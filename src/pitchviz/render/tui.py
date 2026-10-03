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
from pitchviz.data.fetch import search_pitchers, get_pitcher_outings, get_player_names, pitch_data
from pitchviz.data.sequence import at_bat_sequences
from pitchviz.data.filters import (
    pitches_filter,
    pitches_filter_by_at_bat,
    pitches_filter_by_pitch_numbers,
)
from pitchviz.render.builder import VizualizationBuilder

console = Console()

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


def _select_pitcher() -> dict:
    """Returns the confirmed pitcher's match dict (mlbam_id, full_name, team)."""
    while True:
        query = questionary.text("Search for a pitcher:", style=STYLE).ask()
        if query is None:
            sys.exit(0)
        if len(query) < 2:
            console.print("[yellow]Type at least 2 characters.[/yellow]")
            continue

        try:
            matches = search_pitchers(query)
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            continue

        if not matches:
            console.print(f"[yellow]No pitchers found matching '{query}'.[/yellow]")
            continue

        if len(matches) == 1:
            match = matches[0]
        else:
            choice = questionary.select(
                "Select a pitcher:",
                choices=[
                    Choice(
                        title=f"{m['full_name']} ({m['team']})" if m["team"] else m["full_name"],
                        value=m["full_name"],
                    )
                    for m in matches
                ],
                style=STYLE,
            ).ask()
            if choice is None:
                sys.exit(0)
            match = next(m for m in matches if m["full_name"] == choice)

        label = f"{match['full_name']} ({match['team']})" if match["team"] else match["full_name"]
        confirmed = questionary.confirm(f"Use {label}?", default=True, style=STYLE).ask()
        if confirmed is None:
            sys.exit(0)
        if confirmed:
            return match
        # not confirmed — loop back to search


def _select_year() -> int:
    current_year = _date.today().year

    def _validate(text: str) -> bool | str:
        if not text.isdigit() or len(text) != 4:
            return "Enter a 4-digit year."
        year = int(text)
        if not (_MIN_STATCAST_YEAR <= year <= current_year):
            return f"Enter a year between {_MIN_STATCAST_YEAR} and {current_year}."
        return True

    while True:
        answer = questionary.text(
            "Which season?", default=str(current_year), validate=_validate, style=STYLE,
        ).ask()
        if answer is None:
            sys.exit(0)
        year = int(answer)

        confirmed = questionary.confirm(f"Use the {year} season?", default=True, style=STYLE).ask()
        if confirmed is None:
            sys.exit(0)
        if confirmed:
            return year
        # not confirmed — re-prompt for year


def _select_outing(pitcher_name: str, season: int) -> str:
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
        return f"★ {base} (Postseason)" if o["postseason"] else base

    choice = questionary.select(
        f"Select an outing ({pitcher_name}, {season}):",
        choices=[Choice(title=_title(o), value=o["date"]) for o in reversed(outings)],
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
    """Returns a (filter_fn, description, sequential) tuple."""
    mode = questionary.select(
        "What would you like to render?",
        choices=[
            Choice(title="Full outing - every pitch thrown", value="full"),
            Choice(title="A specific at-bat", value="at_bat"),
            Choice(title="Specific pitches from a specific at-bat", value="tunnel"),
        ],
        style=STYLE,
    ).ask()
    if mode is None:
        sys.exit(0)

    if mode == "full":
        return pitches_filter, "Full outing", False

    at_bats = at_bat_sequences(df)
    if not at_bats:
        console.print("[yellow]No at-bats found in this outing.[/yellow]")
        sys.exit(1)

    try:
        batter_names = get_player_names([ab["batter"] for ab in at_bats])
    except Exception:
        batter_names = {}

    ab_choice = questionary.select(
        "Select an at-bat:",
        choices=[
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
        style=STYLE,
    ).ask()
    if ab_choice is None:
        sys.exit(0)

    at_bat = next(ab for ab in at_bats if ab["at_bat_number"] == ab_choice)
    batter_label = batter_names.get(at_bat["batter"], "Unknown batter")

    if mode == "at_bat":
        label = f"vs {batter_label} — Inning {at_bat['inning']}"
        return pitches_filter_by_at_bat(ab_choice, label=label), f"At-Bat #{ab_choice}", True

    # tunnel mode
    pitch_choices = questionary.checkbox(
        "Select 2+ pitches to overlay (tunnel view):",
        choices=[
            Choice(title=_format_pitch_choice(p), value=p["pitch_number"])
            for p in at_bat["pitches"]
        ],
        style=STYLE,
    ).ask()
    if not pitch_choices:
        console.print("[yellow]No pitches selected.[/yellow]")
        sys.exit(1)

    return pitches_filter_by_pitch_numbers(ab_choice, pitch_choices), f"Tunnel view (AB #{ab_choice})", True


def _select_quality() -> str:
    quality = questionary.select("Render quality:", choices=QUALITY_CHOICES, style=STYLE).ask()
    if quality is None:
        sys.exit(0)
    return quality


def run_tui() -> None:
    console.print(Panel.fit("[bold]Pitch Viz[/bold] — interactive renderer", border_style="#67E18D"))

    pitcher = _select_pitcher()
    pitcher_name = pitcher["full_name"]
    year = _select_year()
    date = _select_outing(pitcher_name, year)

    with console.status(f"[bold]Fetching {pitcher_name}'s outing on {date}...[/bold]"):
        try:
            df = pitch_data(start_dt=date, pitcher=pitcher_name)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

    if df.empty:
        console.print(f"[yellow]No pitch data found for {pitcher_name} on {date}.[/yellow]")
        sys.exit(1)

    filt, description, sequential = _select_scope(df)
    quality = _select_quality()

    summary = Table.grid(padding=(0, 2))
    summary.add_row("[bold]Pitcher:[/bold]", pitcher_name)
    summary.add_row("[bold]Date:[/bold]", date)
    summary.add_row("[bold]Scope:[/bold]", description)
    summary.add_row("[bold]Quality:[/bold]", quality)
    console.print(Panel(summary, title="Render Summary", border_style="#1BB999"))

    builder = VizualizationBuilder()
    builder.load_pitches_from_df(df, filt)
    if builder._axes is None:
        console.print("[red]No pitches to render for this selection.[/red]")
        sys.exit(1)
    scene_class = builder.buildm_pitches(sequential=sequential)
    filename = f"{pitcher_name} {date} {description}"

    with console.status("[bold green]Rendering...[/bold green]", spinner="dots"):
        VizualizationBuilder.render(scene_class, quality=quality, filename=filename)

    console.print(Panel(f"[bold green]Done![/bold green] Rendered as '{filename}'", border_style="green"))


if __name__ == "__main__":
    run_tui()
