import pytest


def test_heatmap_unfiltered(monkeypatch, outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: outing_df)

    resp = client.get("/heatmap", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 100
    assert len(body["pitches"]) == 100
    assert body["strike_zone"]["top"] == pytest.approx(3.4710, abs=1e-3)
    assert body["strike_zone"]["bottom"] == pytest.approx(1.6453, abs=1e-3)


def test_heatmap_filters_by_stand(monkeypatch, outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: outing_df)

    resp = client.get("/heatmap", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04", "stand": "L"})

    assert resp.json()["count"] == 88


def test_heatmap_filters_by_pitch_type(monkeypatch, outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: outing_df)

    resp = client.get("/heatmap", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04", "pitch_type": "SL"})

    body = resp.json()
    assert body["count"] == 4
    assert all(p["pitch_type"] == "SL" for p in body["pitches"])


def test_heatmap_empty_outing(monkeypatch, empty_outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: empty_outing_df)

    resp = client.get("/heatmap", params={"pitcher_name": "Paul Skenes", "date": "2024-08-05"})

    assert resp.status_code == 200
    assert resp.json() == {"pitches": [], "count": 0, "strike_zone": None}
