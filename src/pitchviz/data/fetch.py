from concurrent.futures import ThreadPoolExecutor, as_completed

import pybaseball
import pandas as pd
import requests
from rapidfuzz import fuzz, process

pybaseball.cache.enable()

_pitcher_roster_cache: list[dict] | None = None


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


def search_pitchers(query: str) -> list[dict]:
    """Fuzzy-search the active MLB pitcher roster by name, tolerant of typos/partial names."""
    global _pitcher_roster_cache

    if not query or len(query) < 2:
        return []

    if _pitcher_roster_cache is None:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/sports/1/players",
            params={"season": 2026, "gameType": "R"},
            timeout=15,
        )
        response.raise_for_status()
        _pitcher_roster_cache = [
            p for p in response.json().get("people", [])
            if p.get("primaryPosition", {}).get("type") == "Pitcher"
        ]

    names = [p["fullName"] for p in _pitcher_roster_cache]
    matches = process.extract(
        query, names, scorer=fuzz.WRatio, limit=10, score_cutoff=_FUZZY_MATCH_CUTOFF,
    )

    return [
        {
            "mlbam_id":  _pitcher_roster_cache[idx]["id"],
            "full_name": _pitcher_roster_cache[idx]["fullName"],
            "team":      _pitcher_roster_cache[idx].get("currentTeam", {}).get("abbreviation", ""),
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


def get_pitcher_outings(pitcher_name: str, season: int) -> list[dict]:
    """Dates (opponent/home-away/postseason) the named pitcher appeared in during a season.

    Includes both regular-season and postseason outings, merged in chronological order.
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
        }
        for postseason, splits in ((False, regular_splits), (True, postseason_splits))
        for s in splits
    ]

    return sorted(outings, key=lambda o: o["date"])


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
