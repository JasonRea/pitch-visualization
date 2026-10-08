import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pybaseball
import pandas as pd
import numpy as np
import requests
from PIL import Image
from rapidfuzz import fuzz, process

pybaseball.cache.enable()

_pitcher_roster_cache: dict[int, list[dict]] = {}


def _lookup_player_id(pitcher: str) -> int:
    first, last = pitcher.split(" ", 1)
    # Fuzzy search so special chars in names dont give us a hard time
    return int(pybaseball.playerid_lookup(last, first, fuzzy=True)['key_mlbam'].head(1).item())


def pitch_data(start_dt: str, pitcher: str, end_dt: str | None = None) -> pd.DataFrame:
    '''
    start_dt, end_dt -> Date given by YYYY-MM-DD
    pitcher -> First, last of pitcher (eg. Paul Skenes, Tarik Skubal, Huascar Brazoban) really tried to sneak in skubal there
    '''

    try:
        player_id = _lookup_player_id(pitcher)

        if not end_dt:
            end_dt = start_dt

        daily_stats = pybaseball.statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=player_id)

        return daily_stats
    except:
        raise RuntimeError("Failed to retrieve pitch data")


_FUZZY_MATCH_CUTOFF = 55


def search_pitchers(query: str, season: int) -> list[dict]:
    """Fuzzy-search that season's MLB pitcher roster by name, tolerant of typos/partial names.

    Scoped to `season` rather than the current active roster so historical
    pitchers (e.g. retired by now) can still be found when browsing past
    outings — searching against only the current roster would silently miss
    them.
    """
    global _pitcher_roster_cache

    if not query or len(query) < 2:
        return []

    if season not in _pitcher_roster_cache:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/sports/1/players",
            params={"season": season, "gameType": "R"},
            timeout=15,
        )
        response.raise_for_status()
        _pitcher_roster_cache[season] = [
            p for p in response.json().get("people", [])
            if p.get("primaryPosition", {}).get("type") == "Pitcher"
        ]

    roster = _pitcher_roster_cache[season]
    names = [p["fullName"] for p in roster]
    matches = process.extract(
        query, names, scorer=fuzz.WRatio, limit=10, score_cutoff=_FUZZY_MATCH_CUTOFF,
    )

    return [
        {
            "mlbam_id":  roster[idx]["id"],
            "full_name": roster[idx]["fullName"],
            # The roster endpoint's currentTeam object has no "abbreviation"
            # field (confirmed against the real API) — only id/name/link.
            "team":      roster[idx].get("currentTeam", {}).get("name", ""),
        }
        for _, _, idx in matches
    ]


def _fetch_game_log_splits(player_id: int, season: int, game_type: str | None = None) -> list[dict]:
    params = {"stats": "gameLog", "season": season, "group": "pitching", "hydrate": "team"}
    if game_type:
        params["gameType"] = game_type

    response = requests.get(
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats",
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    return (response.json().get("stats") or [{}])[0].get("splits", [])


def _fetch_game_final_statuses(game_pks: list[int]) -> dict[int, bool]:
    """Bulk-resolve gamePks to whether the game has gone Final, in one request.

    A game's gameLog split appears the moment it starts (confirmed live: an
    in-progress game shows up immediately, with Statcast pitch-level data not
    available for it until well after). Missing/unresolvable gamePks default
    to "final" (fail open) rather than block an outing we can't positively
    identify as still in progress.
    """
    if not game_pks:
        return {}

    response = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "gamePks": ",".join(str(pk) for pk in game_pks)},
        timeout=15,
    )
    response.raise_for_status()

    statuses = {}
    for d in response.json().get("dates", []):
        for g in d.get("games", []):
            statuses[g["gamePk"]] = g.get("status", {}).get("abstractGameState") == "Final"
    return statuses


def get_pitcher_outings(pitcher_name: str, season: int) -> list[dict]:
    """Dates (opponent/home-away/postseason/final/game_pk) the named pitcher appeared in during a season.

    Includes both regular-season and postseason outings, merged in chronological order.
    `final` is False for a game that's still in progress (or otherwise not
    yet Final) — such a game may already appear in the game log before any
    Statcast pitch data exists for it; `game_pk` is kept so a non-final
    outing's live data can still be fetched via `live_pitch_data()`.
    """
    try:
        player_id = _lookup_player_id(pitcher_name)
    except Exception:
        raise RuntimeError("Pitcher not found")

    regular_splits = _fetch_game_log_splits(player_id, season)
    postseason_splits = _fetch_game_log_splits(player_id, season, game_type="P")

    outings = [
        {
            "date":        s["date"],
            "opponent":    s.get("opponent", {}).get("abbreviation", ""),
            "home_away":   "home" if s.get("isHome") else "away",
            "postseason":  postseason,
            "game_pk":     s.get("game", {}).get("gamePk"),
        }
        for postseason, splits in ((False, regular_splits), (True, postseason_splits))
        for s in splits
    ]

    try:
        final_statuses = _fetch_game_final_statuses(
            [o["game_pk"] for o in outings if o["game_pk"] is not None]
        )
    except Exception:
        final_statuses = {}

    for o in outings:
        o["final"] = final_statuses.get(o["game_pk"], True)

    return sorted(outings, key=lambda o: o["date"])


def get_pitcher_game_stats(player_id: int, season: int, game_pk: int) -> dict | None:
    """Official box-score pitching line (innings pitched/strikeouts/walks/
    hits/runs/earned runs) for one specific game.

    These are scorer-determined stats (earned runs especially) that aren't
    reliably derivable from raw Statcast pitch rows, so this pulls MLB's own
    already-computed `stat` object off the matching gameLog split — the same
    data get_pitcher_outings() already fetches per-game, just not discarded
    this time. Returns None if no split matches game_pk (e.g. the game isn't
    found in either the regular-season or postseason log).
    """
    regular_splits = _fetch_game_log_splits(player_id, season)
    postseason_splits = _fetch_game_log_splits(player_id, season, game_type="P")

    for split in (*regular_splits, *postseason_splits):
        if split.get("game", {}).get("gamePk") == game_pk:
            stat = split.get("stat", {})
            return {
                "innings_pitched": stat.get("inningsPitched"),  # already "N.1"/"N.2" for partial innings
                "strikeouts":  stat.get("strikeOuts"),
                "walks":       stat.get("baseOnBalls"),
                "hits":        stat.get("hits"),
                "runs":        stat.get("runs"),
                "earned_runs": stat.get("earnedRuns"),
            }

    return None


def _release_point(x0, y0, z0, vx0, vy0, vz0, ax, ay, az, target_y):
    """Extrapolate (x0,y0,z0,...) — defined at the live feed's fixed y=50ft
    reference plane — to the pitcher's actual release point at `target_y`
    (computed by the caller as 60.5 - extension). Solves the same kinematic
    equation `position()` in render/builder.py uses, `y0 + vy0*t +
    0.5*ay*t**2`, for the t at which it equals target_y, then evaluates
    x(t)/z(t) at that t.

    Verified against a real completed game (Blake Snell, 2026-10-04): the
    derived release_pos_x/y/z reproduce pybaseball's published Statcast
    values for that game almost exactly (within ~0.01ft).
    """
    a = 0.5 * ay
    b = vy0
    c = y0 - target_y
    discriminant = b * b - 4 * a * c
    if a == 0 or discriminant < 0:
        return x0, y0, z0  # no solution — fall back to the raw reference-plane point

    sq = math.sqrt(discriminant)
    t1 = (-b + sq) / (2 * a)
    t2 = (-b - sq) / (2 * a)
    # Release is only a fraction of a second from the y=50 reference plane —
    # pick the root closest to zero.
    t = t1 if abs(t1) < abs(t2) else t2

    release_x = x0 + vx0 * t + 0.5 * ax * t * t
    release_z = z0 + vz0 * t + 0.5 * az * t * t
    return release_x, target_y, release_z


_INNING_HALF = {"top": "Top", "bottom": "Bot"}


def live_pitch_data(game_pk: int, pitcher: str) -> pd.DataFrame:
    """Pitch-by-pitch data for one pitcher in one (possibly still in-progress)
    game, sourced from MLB's live game feed rather than pybaseball/Statcast.

    Confirmed against a completed game that this is the same underlying
    measurement pipeline (vx0/vz0/ax/ay/az matched pybaseball's Statcast
    output exactly) — it's just available immediately, without Baseball
    Savant's batch-processing delay. Returns the same columns pitch_data()
    does, so it's a drop-in substitute wherever pitch_data() is used.
    """
    response = requests.get(
        f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
        timeout=15,
    )
    response.raise_for_status()
    plays = response.json().get("liveData", {}).get("plays", {}).get("allPlays", [])

    rows = []
    for play in plays:
        matchup = play.get("matchup", {})
        if matchup.get("pitcher", {}).get("fullName") != pitcher:
            continue

        about = play.get("about", {})
        at_bat_number = about.get("atBatIndex", 0) + 1
        inning = about.get("inning")
        inning_topbot = _INNING_HALF.get(about.get("halfInning"), "")
        stand = matchup.get("batSide", {}).get("code")
        batter = matchup.get("batter", {}).get("id")
        # Only attach the at-bat's outcome once it's actually concluded —
        # for the at-bat currently in progress we don't know it yet.
        final_event = play.get("result", {}).get("eventType") if about.get("isComplete") else None

        pitch_events = [e for e in play.get("playEvents", []) if e.get("isPitch")]
        balls, strikes = 0, 0  # pre-pitch count, resets each at-bat
        for i, event in enumerate(pitch_events):
            pdata = event.get("pitchData", {})
            coords = pdata.get("coordinates", {})
            extension = pdata.get("extension")
            if not coords or extension is None:
                continue

            release_x, release_y, release_z = _release_point(
                coords["x0"], coords["y0"], coords["z0"],
                coords["vX0"], coords["vY0"], coords["vZ0"],
                coords["aX"], coords["aY"], coords["aZ"],
                target_y=60.5 - extension,
            )

            details = event.get("details", {})
            rows.append({
                "vx0": coords["vX0"], "vy0": coords["vY0"], "vz0": coords["vZ0"],
                "ax": coords["aX"], "ay": coords["aY"], "az": coords["aZ"],
                "release_pos_x": release_x, "release_pos_y": release_y, "release_pos_z": release_z,
                "pitch_type":    details.get("type", {}).get("code"),
                "pitch_name":    details.get("type", {}).get("description"),
                "release_speed": pdata.get("startSpeed"),
                "type":          details.get("code"),
                "description":   details.get("description"),
                "zone":          pdata.get("zone"),
                "plate_x":       coords.get("pX"),
                "plate_z":       coords.get("pZ"),
                "sz_top":        pdata.get("strikeZoneTop"),
                "sz_bot":        pdata.get("strikeZoneBottom"),
                "stand":         stand,
                "batter":        batter,
                "inning":        inning,
                "inning_topbot": inning_topbot,
                "at_bat_number": at_bat_number,
                "pitch_number":  event.get("pitchNumber"),
                "balls":         balls,
                "strikes":       strikes,
                "events":        final_event if i == len(pitch_events) - 1 else None,
            })

            count = event.get("count", {})
            balls = count.get("balls", balls)
            strikes = count.get("strikes", strikes)

    return pd.DataFrame(rows)


def get_player_names(player_ids: list[int]) -> dict[int, str]:
    """Bulk-resolve player MLBAM IDs to full names in a single request."""
    unique_ids = sorted(set(player_ids))
    if not unique_ids:
        return {}

    response = requests.get(
        "https://statsapi.mlb.com/api/v1/people",
        params={"personIds": ",".join(str(i) for i in unique_ids)},
        timeout=15,
    )
    response.raise_for_status()

    return {p["id"]: p["fullName"] for p in response.json().get("people", [])}


_HEIGHT_RE = re.compile(r"(\d+)'\s*(\d+)\"?")


def get_player_heights(player_ids: list[int]) -> dict[int, float]:
    """Bulk-resolve player MLBAM IDs to height in inches, for the ABS strike zone formula.

    Same endpoint as get_player_names (confirmed live: every person object
    already includes a "height" field like "6' 6\"", no extra hydrate
    needed). IDs with a missing or unparseable height are simply omitted
    rather than raising.
    """
    unique_ids = sorted(set(player_ids))
    if not unique_ids:
        return {}

    response = requests.get(
        "https://statsapi.mlb.com/api/v1/people",
        params={"personIds": ",".join(str(i) for i in unique_ids)},
        timeout=15,
    )
    response.raise_for_status()

    heights = {}
    for p in response.json().get("people", []):
        match = _HEIGHT_RE.match(p.get("height", ""))
        if match:
            feet, inches = match.groups()
            heights[p["id"]] = int(feet) * 12 + int(inches)
    return heights


def get_player_headshot(player_id: int) -> np.ndarray | None:
    """Fetch a player's MLB headshot as an RGBA numpy array, ready for
    Manim's ImageMobject (confirmed it accepts a numpy array directly, no
    temp file needed).

    Ported from this project's pre-refactor fetch_data.py (commit e50f7bd),
    which returned a PIL Image from the same URL — re-verified live before
    porting. Returns None on any failure (bad status, unparseable image,
    network error) rather than raising, so a headshot hiccup never blocks
    a render.
    """
    url = (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "d_people:generic:headshot:67:current.png/w_640,q_auto:best/"
        f"v1/people/{player_id}/headshot/silo/current.png"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        return np.array(image)
    except Exception:
        return None


def get_game_ids(date: str) -> list[int]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}"
    response = requests.get(url).json()

    return [
        game["gamePk"]
        for date_entry in response.get("dates", [])
        for game in date_entry.get("games", [])
    ]

def daily_pitches(date: str) -> pd.DataFrame:
    game_ids = get_game_ids(date)

    with ThreadPoolExecutor(max_workers=len(game_ids)) as executor:
        futures = {executor.submit(pybaseball.statcast_single_game, gid): gid for gid in game_ids}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Failed for game {futures[future]}: {e}")

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
