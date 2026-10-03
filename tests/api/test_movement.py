import pytest


def test_movement_golden_values(monkeypatch, outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: outing_df)

    resp = client.get("/movement", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["pitches"]) == 100

    ff = next(s for s in body["summary"] if s["code"] == "FF")
    assert ff["count"] == 40
    assert ff["avg_velo"] == pytest.approx(98.0425)
    assert ff["avg_hb_in"] == pytest.approx(-14.214)

    assert sum(s["count"] for s in body["summary"]) == 100


def test_movement_empty_outing(monkeypatch, empty_outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: empty_outing_df)

    resp = client.get("/movement", params={"pitcher_name": "Paul Skenes", "date": "2024-08-05"})

    assert resp.status_code == 200
    assert resp.json() == {"pitches": [], "summary": []}
