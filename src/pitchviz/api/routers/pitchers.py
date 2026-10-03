import httpx
from fastapi import APIRouter

from pitchviz.api.state import players_cache

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /pitchers?q={name}
# ---------------------------------------------------------------------------

@router.get("/pitchers")
async def search_pitchers(q: str = ""):
    if not q or len(q) < 2:
        return []

    if "roster" not in players_cache:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://statsapi.mlb.com/api/v1/sports/1/players",
                params={"season": 2026, "gameType": "R"},
                timeout=15,
            )
            r.raise_for_status()
        players_cache["roster"] = [
            p for p in r.json().get("people", [])
            if p.get("primaryPosition", {}).get("type") == "Pitcher"
        ]

    q_lower = q.lower()
    return [
        {
            "mlbam_id":  p["id"],
            "full_name": p["fullName"],
            "team":      p.get("currentTeam", {}).get("abbreviation", ""),
        }
        for p in players_cache["roster"]
        if q_lower in p["fullName"].lower()
    ][:10]
