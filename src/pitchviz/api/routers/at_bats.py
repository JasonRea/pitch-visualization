from fastapi import APIRouter

from pitchviz.api.deps import get_outing_df
from pitchviz.data.sequence import at_bat_sequences

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /at-bats?pitcher_name={name}&date={YYYY-MM-DD}
# ---------------------------------------------------------------------------

@router.get("/at-bats")
async def get_at_bats(pitcher_name: str, date: str):
    df = await get_outing_df(pitcher_name, date)

    if df.empty:
        return []

    return at_bat_sequences(df)
