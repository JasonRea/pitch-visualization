import asyncio

from fastapi import APIRouter, HTTPException

from pitchviz.config import PITCH_COLORS, PITCH_NAMES
from pitchviz.data.fetch import pitch_data
from pitchviz.api.state import pitch_type_cache

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /pitch-types?pitcher_name={name}&date={YYYY-MM-DD}
# ---------------------------------------------------------------------------

@router.get("/pitch-types")
async def get_pitch_types(pitcher_name: str, date: str):
    cache_key = f"{pitcher_name}:{date}"
    if cache_key in pitch_type_cache:
        return pitch_type_cache[cache_key]

    loop = asyncio.get_event_loop()

    def _fetch() -> list[dict]:
        df = pitch_data(date, pitcher_name)
        if df.empty:
            return []
        rows = (
            df[["pitch_type", "pitch_name"]]
            .dropna()
            .drop_duplicates("pitch_type")
        )
        result = []
        for _, row in rows.iterrows():
            code = row["pitch_type"]
            result.append({
                "code":  code,
                "name":  row.get("pitch_name") or PITCH_NAMES.get(code, code),
                "color": PITCH_COLORS.get(code, "#9C8975"),
                "count": int((df["pitch_type"] == code).sum()),
            })
        result.sort(key=lambda x: -x["count"])
        return result

    try:
        result = await loop.run_in_executor(None, _fetch)
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to fetch pitch data: {e}")

    pitch_type_cache[cache_key] = result
    return result
