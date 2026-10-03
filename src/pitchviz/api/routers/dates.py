import asyncio

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /dates?pitcher_name={name}&season={year}
# ---------------------------------------------------------------------------

@router.get("/dates")
async def get_dates(pitcher_name: str, season: int = 2026):
    loop = asyncio.get_event_loop()

    def _lookup_id() -> int:
        import pybaseball
        parts = pitcher_name.split(" ", 1)
        first, last = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
        result = pybaseball.playerid_lookup(last, first, fuzzy=True)
        if result.empty:
            raise ValueError("not found")
        return int(result["key_mlbam"].head(1).item())

    try:
        player_id = await loop.run_in_executor(None, _lookup_id)
    except ValueError:
        raise HTTPException(404, detail="Pitcher not found")
    except Exception:
        raise HTTPException(500, detail="Failed to look up pitcher")

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats",
            params={"stats": "gameLog", "season": season, "group": "pitching"},
            timeout=15,
        )
        r.raise_for_status()

    splits = (r.json().get("stats") or [{}])[0].get("splits", [])
    return [
        {
            "date":      s["date"],
            "opponent":  s.get("opponent", {}).get("abbreviation", ""),
            "home_away": "home" if s.get("isHome") else "away",
        }
        for s in splits
    ]
