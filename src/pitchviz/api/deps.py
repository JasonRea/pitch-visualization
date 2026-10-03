import asyncio

import pandas as pd
from fastapi import HTTPException

from pitchviz.data.fetch import pitch_data
from pitchviz.api.state import outing_cache


async def get_outing_df(pitcher_name: str, date: str) -> pd.DataFrame:
    """Fetch (or return cached) raw per-pitch Statcast data for one outing.

    Shared by /pitch-types, /movement, /heatmap, and /at-bats so requesting
    several of them for the same pitcher+date only pays the Statcast round
    trip (3-8s) once.
    """
    cache_key = f"{pitcher_name}:{date}"
    if cache_key in outing_cache:
        return outing_cache[cache_key]

    loop = asyncio.get_event_loop()
    try:
        df = await loop.run_in_executor(None, pitch_data, date, pitcher_name)
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to fetch pitch data: {e}")

    outing_cache[cache_key] = df
    return df
