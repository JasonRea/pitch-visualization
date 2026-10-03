from fastapi import APIRouter

from pitchviz.api.deps import get_outing_df
from pitchviz.data.stats import movement_rows, movement_summary

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /movement?pitcher_name={name}&date={YYYY-MM-DD}
# ---------------------------------------------------------------------------

@router.get("/movement")
async def get_movement(pitcher_name: str, date: str):
    df = await get_outing_df(pitcher_name, date)

    if df.empty:
        return {"pitches": [], "summary": []}

    return {
        "pitches": movement_rows(df),
        "summary": movement_summary(df),
    }
