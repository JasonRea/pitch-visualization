import re

import pandas as pd
import respx
from httpx import Response

GAMELOG_URL = re.compile(r"https://statsapi\.mlb\.com/api/v1/people/\d+/stats")


def _gamelog_payload():
    return {
        "stats": [
            {
                "splits": [
                    {"date": "2024-08-04", "opponent": {"abbreviation": "CHC"}, "isHome": True},
                    {"date": "2024-08-10", "opponent": {"abbreviation": "STL"}, "isHome": False},
                ]
            }
        ]
    }


@respx.mock
def test_get_dates_returns_game_log(monkeypatch, client):
    monkeypatch.setattr(
        "pybaseball.playerid_lookup",
        lambda *a, **k: pd.DataFrame({"key_mlbam": [694973]}),
    )
    respx.get(GAMELOG_URL).mock(return_value=Response(200, json=_gamelog_payload()))

    resp = client.get("/dates", params={"pitcher_name": "Paul Skenes", "season": 2024})

    assert resp.status_code == 200
    body = resp.json()
    assert body == [
        {"date": "2024-08-04", "opponent": "CHC", "home_away": "home"},
        {"date": "2024-08-10", "opponent": "STL", "home_away": "away"},
    ]


def test_get_dates_pitcher_not_found_returns_404(monkeypatch, client):
    monkeypatch.setattr(
        "pybaseball.playerid_lookup",
        lambda *a, **k: pd.DataFrame({"key_mlbam": []}),
    )

    resp = client.get("/dates", params={"pitcher_name": "Nobody Real"})

    assert resp.status_code == 404


def test_get_dates_lookup_failure_returns_500(monkeypatch, client):
    def boom(*a, **k):
        raise RuntimeError("statsapi down")

    monkeypatch.setattr("pybaseball.playerid_lookup", boom)

    resp = client.get("/dates", params={"pitcher_name": "Paul Skenes"})

    assert resp.status_code == 500
