import pandas as pd
from fastapi import APIRouter

from pitchviz.api.deps import get_outing_df
from pitchviz.data.filters import location_filter

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /heatmap?pitcher_name={name}&date={YYYY-MM-DD}&pitch_type={code}&stand={L|R}
# ---------------------------------------------------------------------------

@router.get("/heatmap")
async def get_heatmap(pitcher_name: str, date: str, pitch_type: str = "", stand: str = ""):
    df = await get_outing_df(pitcher_name, date)

    if df.empty:
        return {"pitches": [], "count": 0, "strike_zone": None}

    located = location_filter(df, pitch_type=pitch_type or None, stand=stand or None)

    if located.empty:
        return {"pitches": [], "count": 0, "strike_zone": None}

    pitches = [
        {
            "pitch_type":  row["pitch_type"],
            "plate_x":     float(row["plate_x"]),
            "plate_z":     float(row["plate_z"]),
            "zone":        None if pd.isna(row["zone"]) else int(row["zone"]),
            "stand":       row["stand"],
            "description": row["description"],
        }
        for _, row in located.iterrows()
    ]

    sz_top = located["sz_top"].mean()
    sz_bot = located["sz_bot"].mean()

    return {
        "pitches": pitches,
        "count": len(pitches),
        "strike_zone": None if pd.isna(sz_top) or pd.isna(sz_bot) else {
            "top":    float(sz_top),
            "bottom": float(sz_bot),
        },
    }
