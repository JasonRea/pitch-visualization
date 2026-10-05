import pandas as pd
import pytest

import pitchviz.data.fetch as fetch_module
from pitchviz.data.fetch import pitch_data


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_pitch_data_success(monkeypatch, outing_df):
    lookup_result = pd.DataFrame({"key_mlbam": [694973]})
    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: lookup_result)
    monkeypatch.setattr("pybaseball.statcast_pitcher", lambda *a, **k: outing_df)

    result = pitch_data(start_dt="2024-08-04", pitcher="Paul Skenes")

    assert len(result) == len(outing_df)


def test_pitch_data_defaults_end_dt_to_start_dt(monkeypatch, outing_df):
    lookup_result = pd.DataFrame({"key_mlbam": [694973]})
    seen = {}

    def fake_statcast_pitcher(start_dt, end_dt, player_id):
        seen["start_dt"] = start_dt
        seen["end_dt"] = end_dt
        return outing_df

    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: lookup_result)
    monkeypatch.setattr("pybaseball.statcast_pitcher", fake_statcast_pitcher)

    pitch_data(start_dt="2024-08-04", pitcher="Paul Skenes")

    assert seen["start_dt"] == "2024-08-04"
    assert seen["end_dt"] == "2024-08-04"


def test_pitch_data_wraps_any_failure_as_runtime_error(monkeypatch):
    def boom(*a, **k):
        raise ValueError("network exploded")

    monkeypatch.setattr("pybaseball.playerid_lookup", boom)

    with pytest.raises(RuntimeError, match="Failed to retrieve pitch data"):
        pitch_data(start_dt="2024-08-04", pitcher="Paul Skenes")


def test_search_pitchers_requires_min_two_chars():
    assert fetch_module.search_pitchers("s", season=2024) == []


def test_search_pitchers_filters_by_name_and_caches(monkeypatch):
    monkeypatch.setattr(fetch_module, "_pitcher_roster_cache", {})
    roster_response = _FakeResponse({
        "people": [
            {"id": 1, "fullName": "Paul Skenes", "primaryPosition": {"type": "Pitcher"}, "currentTeam": {"name": "Pittsburgh Pirates"}},
            {"id": 2, "fullName": "Shohei Ohtani", "primaryPosition": {"type": "Outfielder"}, "currentTeam": {"name": "Los Angeles Dodgers"}},
        ]
    })
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(args)
        return roster_response

    monkeypatch.setattr("requests.get", fake_get)

    result = fetch_module.search_pitchers("skenes", season=2024)
    assert result == [{"mlbam_id": 1, "full_name": "Paul Skenes", "team": "Pittsburgh Pirates"}]

    # second call for the same season should hit the in-process cache, not requests.get again
    fetch_module.search_pitchers("skenes", season=2024)
    assert len(calls) == 1


def test_search_pitchers_caches_per_season(monkeypatch):
    monkeypatch.setattr(fetch_module, "_pitcher_roster_cache", {})
    roster_response = _FakeResponse({
        "people": [
            {"id": 1, "fullName": "Paul Skenes", "primaryPosition": {"type": "Pitcher"}, "currentTeam": {"name": "Pittsburgh Pirates"}},
        ]
    })
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(args)
        return roster_response

    monkeypatch.setattr("requests.get", fake_get)

    fetch_module.search_pitchers("skenes", season=2023)
    fetch_module.search_pitchers("skenes", season=2024)

    # different seasons are independent cache entries, so both hit the network
    assert len(calls) == 2
    assert set(fetch_module._pitcher_roster_cache.keys()) == {2023, 2024}


def test_search_pitchers_tolerates_typos(monkeypatch):
    monkeypatch.setattr(fetch_module, "_pitcher_roster_cache", {})
    roster_response = _FakeResponse({
        "people": [
            {"id": 1, "fullName": "Paul Skenes", "primaryPosition": {"type": "Pitcher"}, "currentTeam": {"name": "Pittsburgh Pirates"}},
        ]
    })
    monkeypatch.setattr("requests.get", lambda *a, **k: roster_response)

    result = fetch_module.search_pitchers("Paul Skeens", season=2024)  # transposed typo

    assert result == [{"mlbam_id": 1, "full_name": "Paul Skenes", "team": "Pittsburgh Pirates"}]


def _gamelog_response(date, opponent, is_home):
    return _FakeResponse({
        "stats": [{"splits": [
            {"date": date, "opponent": {"abbreviation": opponent}, "isHome": is_home},
        ]}]
    })


def test_get_pitcher_outings_requests_team_hydration(monkeypatch):
    """Regression test: the opponent object has no `abbreviation` field unless
    `hydrate=team` is requested (confirmed against the real MLB Stats API)."""
    lookup_result = pd.DataFrame({"key_mlbam": [694973]})
    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: lookup_result)
    seen_params = []

    def fake_get(url, params=None, **kwargs):
        seen_params.append(params)
        return _gamelog_response("2024-08-04", "CIN", True)

    monkeypatch.setattr("requests.get", fake_get)

    fetch_module.get_pitcher_outings("Paul Skenes", 2024)

    assert all(p["hydrate"] == "team" for p in seen_params)


def test_get_pitcher_outings_returns_regular_season_game_log(monkeypatch):
    lookup_result = pd.DataFrame({"key_mlbam": [694973]})
    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: lookup_result)

    def fake_get(url, params=None, **kwargs):
        if params and params.get("gameType") == "P":
            return _FakeResponse({"stats": [{"splits": []}]})
        return _gamelog_response("2024-08-04", "CIN", True)

    monkeypatch.setattr("requests.get", fake_get)

    result = fetch_module.get_pitcher_outings("Paul Skenes", 2024)

    assert result == [{"date": "2024-08-04", "opponent": "CIN", "home_away": "home", "postseason": False}]


def test_get_pitcher_outings_merges_and_tags_postseason(monkeypatch):
    lookup_result = pd.DataFrame({"key_mlbam": [656427]})
    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: lookup_result)

    def fake_get(url, params=None, **kwargs):
        if params and params.get("gameType") == "P":
            return _gamelog_response("2024-10-06", "SD", True)
        return _gamelog_response("2024-09-25", "COL", False)

    monkeypatch.setattr("requests.get", fake_get)

    result = fetch_module.get_pitcher_outings("Jack Flaherty", 2024)

    assert result == [
        {"date": "2024-09-25", "opponent": "COL", "home_away": "away", "postseason": False},
        {"date": "2024-10-06", "opponent": "SD", "home_away": "home", "postseason": True},
    ]


def test_get_pitcher_outings_wraps_lookup_failure_as_runtime_error(monkeypatch):
    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: pd.DataFrame({"key_mlbam": []}))

    with pytest.raises(RuntimeError, match="Pitcher not found"):
        fetch_module.get_pitcher_outings("Nobody Here", 2024)


def test_get_player_names_returns_empty_dict_for_no_ids():
    assert fetch_module.get_player_names([]) == {}


def test_get_player_names_resolves_bulk_ids(monkeypatch):
    seen = {}

    def fake_get(url, params=None, **kwargs):
        seen["params"] = params
        return _FakeResponse({
            "people": [
                {"id": 682998, "fullName": "Corbin Carroll"},
                {"id": 656427, "fullName": "Jack Flaherty"},
            ]
        })

    monkeypatch.setattr("requests.get", fake_get)

    result = fetch_module.get_player_names([682998, 656427, 682998])

    assert result == {682998: "Corbin Carroll", 656427: "Jack Flaherty"}
    assert seen["params"]["personIds"] == "656427,682998"


def test_get_player_heights_returns_empty_dict_for_no_ids():
    assert fetch_module.get_player_heights([]) == {}


def test_get_player_heights_parses_feet_and_inches(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse({
        "people": [
            {"id": 694973, "height": "6' 6\""},
            {"id": 656427, "height": "6' 4\""},
        ]
    }))

    result = fetch_module.get_player_heights([694973, 656427])

    assert result == {694973: 78, 656427: 76}


def test_get_player_heights_omits_missing_or_unparseable_height(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse({
        "people": [
            {"id": 1, "height": "6' 2\""},
            {"id": 2},
            {"id": 3, "height": "unknown"},
        ]
    }))

    result = fetch_module.get_player_heights([1, 2, 3])

    assert result == {1: 74}
