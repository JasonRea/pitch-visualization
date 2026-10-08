from io import BytesIO

import pandas as pd
import pytest
from PIL import Image

import pitchviz.data.fetch as fetch_module
from pitchviz.data.fetch import pitch_data


class _FakeImageResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


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


def _gamelog_response(date, opponent, is_home, game_pk=None):
    return _FakeResponse({
        "stats": [{"splits": [
            {"date": date, "opponent": {"abbreviation": opponent}, "isHome": is_home,
             "game": {"gamePk": game_pk}},
        ]}]
    })


def _schedule_response(statuses: dict):
    """statuses: {gamePk: abstractGameState}"""
    return _FakeResponse({
        "dates": [{"games": [
            {"gamePk": pk, "status": {"abstractGameState": state}}
            for pk, state in statuses.items()
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

    assert result == [{"date": "2024-08-04", "opponent": "CIN", "home_away": "home", "postseason": False, "game_pk": None, "final": True}]


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
        {"date": "2024-09-25", "opponent": "COL", "home_away": "away", "postseason": False, "game_pk": None, "final": True},
        {"date": "2024-10-06", "opponent": "SD", "home_away": "home", "postseason": True, "game_pk": None, "final": True},
    ]


def test_get_pitcher_outings_marks_in_progress_game_as_not_final(monkeypatch):
    lookup_result = pd.DataFrame({"key_mlbam": [668909]})
    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: lookup_result)

    def fake_get(url, params=None, **kwargs):
        if "schedule" in url:
            return _schedule_response({111: "Final", 222: "Live"})
        if params and params.get("gameType") == "P":
            return _FakeResponse({"stats": [{"splits": []}]})
        return _FakeResponse({"stats": [{"splits": [
            {"date": "2026-10-04", "opponent": {"abbreviation": "NYY"}, "isHome": True, "game": {"gamePk": 111}},
            {"date": "2026-10-05", "opponent": {"abbreviation": "TB"}, "isHome": False, "game": {"gamePk": 222}},
        ]}]})

    monkeypatch.setattr("requests.get", fake_get)

    result = fetch_module.get_pitcher_outings("Gavin Williams", 2026)

    assert result == [
        {"date": "2026-10-04", "opponent": "NYY", "home_away": "home", "postseason": False, "game_pk": 111, "final": True},
        {"date": "2026-10-05", "opponent": "TB", "home_away": "away", "postseason": False, "game_pk": 222, "final": False},
    ]


def test_get_pitcher_outings_wraps_lookup_failure_as_runtime_error(monkeypatch):
    monkeypatch.setattr("pybaseball.playerid_lookup", lambda *a, **k: pd.DataFrame({"key_mlbam": []}))

    with pytest.raises(RuntimeError, match="Pitcher not found"):
        fetch_module.get_pitcher_outings("Nobody Here", 2024)


def test_get_pitcher_game_stats_returns_box_score_for_matching_game(monkeypatch):
    def fake_get(url, params=None, **kwargs):
        if params and params.get("gameType") == "P":
            return _FakeResponse({"stats": [{"splits": []}]})
        return _FakeResponse({"stats": [{"splits": [
            {
                "date": "2024-08-04", "game": {"gamePk": 745468},
                "stat": {
                    "inningsPitched": "6.1", "strikeOuts": 11, "baseOnBalls": 1,
                    "hits": 2, "runs": 1, "earnedRuns": 1,
                },
            },
        ]}]})

    monkeypatch.setattr("requests.get", fake_get)

    result = fetch_module.get_pitcher_game_stats(player_id=694973, season=2024, game_pk=745468)

    assert result == {
        "innings_pitched": "6.1", "strikeouts": 11, "walks": 1,
        "hits": 2, "runs": 1, "earned_runs": 1,
    }


def test_get_pitcher_game_stats_returns_none_when_game_not_found(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse({"stats": [{"splits": []}]}))

    result = fetch_module.get_pitcher_game_stats(player_id=694973, season=2024, game_pk=999999)

    assert result is None


def test_get_player_headshot_returns_rgba_array_on_success(monkeypatch):
    buf = BytesIO()
    Image.new("RGB", (4, 6), color=(10, 20, 30)).save(buf, format="PNG")
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeImageResponse(buf.getvalue()))

    result = fetch_module.get_player_headshot(694973)

    assert result is not None
    assert result.shape == (6, 4, 4)  # height, width, RGBA


def test_get_player_headshot_returns_none_on_http_failure(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeImageResponse(b"", status_code=404))

    assert fetch_module.get_player_headshot(1) is None


def test_get_player_headshot_returns_none_on_unparseable_content(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeImageResponse(b"not an image"))

    assert fetch_module.get_player_headshot(1) is None


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


def test_release_point_matches_real_statcast_release_point():
    # Real pitchData.coordinates + extension pulled from a live game feed
    # (Blake Snell, 2026-10-04); expected output verified against that same
    # game's actual pybaseball/Statcast release_pos_x/y/z (within ~0.01ft).
    rx, ry, rz = fetch_module._release_point(
        x0=-3.2163147089092776, y0=50.00590460983415, z0=5.2300425264242545,
        vx0=15.35355128271588, vy0=-140.61221572217386, vz0=-7.871861852615233,
        ax=-9.079836994484374, ay=35.225089200189, az=-7.785828980055117,
        target_y=60.5 - 6.770410849286073,
    )

    assert rx == pytest.approx(-3.625, abs=0.01)
    assert ry == pytest.approx(53.730, abs=0.01)
    assert rz == pytest.approx(5.435, abs=0.01)


def _live_feed_response(pitcher_name="Blake Snell", pitcher_id=621107):
    return _FakeResponse({
        "liveData": {
            "plays": {
                "allPlays": [
                    {
                        "about": {"atBatIndex": 0, "inning": 1, "halfInning": "top", "isComplete": True},
                        "matchup": {
                            "batter": {"id": 700250, "fullName": "Ben Rice"},
                            "batSide": {"code": "L"},
                            "pitcher": {"id": pitcher_id, "fullName": pitcher_name},
                        },
                        "result": {"event": "Flyout", "eventType": "field_out"},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "count": {"balls": 1, "strikes": 0},
                                "details": {
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "code": "B", "description": "Ball",
                                },
                                "pitchData": {
                                    "startSpeed": 97.3, "zone": 14,
                                    "strikeZoneTop": 3.281, "strikeZoneBottom": 1.656,
                                    "extension": 6.770410849286073,
                                    "coordinates": {
                                        "x0": -3.2163147089092776, "y0": 50.00590460983415, "z0": 5.2300425264242545,
                                        "vX0": 15.35355128271588, "vY0": -140.61221572217386, "vZ0": -7.871861852615233,
                                        "aX": -9.079836994484374, "aY": 35.225089200189, "aZ": -7.785828980055117,
                                        "pX": 1.746, "pZ": 1.871,
                                    },
                                },
                            },
                            {
                                "isPitch": True,
                                "pitchNumber": 2,
                                "count": {"balls": 1, "strikes": 1},
                                "details": {
                                    "type": {"code": "SL", "description": "Slider"},
                                    "code": "S", "description": "Called Strike",
                                },
                                "pitchData": {
                                    "startSpeed": 88.1, "zone": 5,
                                    "strikeZoneTop": 3.281, "strikeZoneBottom": 1.656,
                                    "extension": 6.5,
                                    "coordinates": {
                                        "x0": -3.1, "y0": 50.0, "z0": 5.1,
                                        "vX0": 10.0, "vY0": -130.0, "vZ0": -6.0,
                                        "aX": -8.0, "aY": 30.0, "aZ": -10.0,
                                        "pX": 0.1, "pZ": 2.3,
                                    },
                                },
                            },
                        ],
                    },
                ],
            },
        },
    })


def test_live_pitch_data_maps_pitches_for_the_requested_pitcher(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _live_feed_response())

    df = fetch_module.live_pitch_data(game_pk=849823, pitcher="Blake Snell")

    assert len(df) == 2
    assert list(df["pitch_type"]) == ["FF", "SL"]
    assert list(df["pitch_number"]) == [1, 2]
    assert list(df["at_bat_number"]) == [1, 1]  # atBatIndex 0 -> at_bat_number 1
    assert list(df["batter"]) == [700250, 700250]
    assert list(df["stand"]) == ["L", "L"]
    assert list(df["inning"]) == [1, 1]
    assert list(df["inning_topbot"]) == ["Top", "Top"]
    # pre-pitch count: first pitch of the at-bat is always 0-0
    assert list(df["balls"]) == [0, 1]
    assert list(df["strikes"]) == [0, 0]
    # only the last pitch of a completed at-bat carries the outcome
    assert pd.isna(df["events"].iloc[0])
    assert df["events"].iloc[1] == "field_out"


def test_live_pitch_data_filters_to_the_requested_pitcher(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _live_feed_response())

    df = fetch_module.live_pitch_data(game_pk=849823, pitcher="Someone Else")

    assert df.empty


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
