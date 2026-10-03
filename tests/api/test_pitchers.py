import respx
from httpx import Response

ROSTER_URL = "https://statsapi.mlb.com/api/v1/sports/1/players"


def _roster_payload():
    return {
        "people": [
            {"id": 1, "fullName": "Paul Skenes", "primaryPosition": {"type": "Pitcher"}, "currentTeam": {"abbreviation": "PIT"}},
            {"id": 2, "fullName": "Spencer Strider", "primaryPosition": {"type": "Pitcher"}, "currentTeam": {"abbreviation": "ATL"}},
            {"id": 3, "fullName": "Not A Pitcher", "primaryPosition": {"type": "Catcher"}, "currentTeam": {"abbreviation": "NYY"}},
        ]
    }


@respx.mock
def test_search_pitchers_filters_by_name_case_insensitive(client):
    respx.get(ROSTER_URL).mock(return_value=Response(200, json=_roster_payload()))

    resp = client.get("/pitchers", params={"q": "skenes"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0] == {"mlbam_id": 1, "full_name": "Paul Skenes", "team": "PIT"}


@respx.mock
def test_search_pitchers_excludes_non_pitchers(client):
    respx.get(ROSTER_URL).mock(return_value=Response(200, json=_roster_payload()))

    resp = client.get("/pitchers", params={"q": "a"})

    names = {p["full_name"] for p in resp.json()}
    assert "Not A Pitcher" not in names


def test_search_pitchers_requires_min_2_chars(client):
    resp = client.get("/pitchers", params={"q": "s"})

    assert resp.status_code == 200
    assert resp.json() == []


@respx.mock
def test_search_pitchers_caches_roster_across_calls(client):
    route = respx.get(ROSTER_URL).mock(return_value=Response(200, json=_roster_payload()))

    client.get("/pitchers", params={"q": "skenes"})
    client.get("/pitchers", params={"q": "strider"})

    assert route.call_count == 1
