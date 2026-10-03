def test_at_bats_golden_values(monkeypatch, outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: outing_df)

    resp = client.get("/at-bats", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 22

    numbers = [ab["at_bat_number"] for ab in body]
    assert numbers == sorted(numbers)

    first = body[0]
    assert first["final_outcome"] == "Single"
    assert first["pitches"][0]["balls"] == 0
    assert first["pitches"][0]["strikes"] == 0

    assert sum(len(ab["pitches"]) for ab in body) == 100


def test_at_bats_empty_outing_returns_empty_list(monkeypatch, empty_outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: empty_outing_df)

    resp = client.get("/at-bats", params={"pitcher_name": "Paul Skenes", "date": "2024-08-05"})

    assert resp.status_code == 200
    assert resp.json() == []
