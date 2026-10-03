import pandas as pd
import pytest

from pitchviz.data.fetch import pitch_data


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
