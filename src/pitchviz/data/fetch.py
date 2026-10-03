from concurrent.futures import ThreadPoolExecutor, as_completed

import pybaseball
import pandas as pd
import requests

pybaseball.cache.enable()


def pitch_data(start_dt: str, pitcher: str, end_dt: str | None = None) -> pd.DataFrame:
    '''
    start_dt, end_dt -> Date given by YYYY-MM-DD
    pitcher -> First, last of pitcher (eg. Paul Skenes, Tarik Skubal, Huascar Brazoban) really tried to sneak in skubal there
    '''

    try:
        first, last = pitcher.split(" ")

        # Fuzzy search so special chars in names dont give us a hard time
        player_id = pybaseball.playerid_lookup(last, first, fuzzy=True)['key_mlbam'].head(1).item()

        if not end_dt:
            end_dt = start_dt

        daily_stats = pybaseball.statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=player_id)

        return daily_stats
    except:
        raise RuntimeError("Failed to retrieve pitch data")

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
