def test_pitch_types_golden_counts(monkeypatch, outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: outing_df)

    resp = client.get("/pitch-types", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04"})

    assert resp.status_code == 200
    body = resp.json()
    counts = {row["code"]: row["count"] for row in body}
    assert counts == {"FF": 40, "FS": 30, "CU": 19, "CH": 5, "SL": 4, "ST": 2}
    assert sum(counts.values()) == 100
    # sorted descending by count
    assert [row["count"] for row in body] == sorted(counts.values(), reverse=True)


def test_pitch_types_empty_outing_returns_empty_list(monkeypatch, empty_outing_df, client):
    monkeypatch.setattr("pitchviz.api.deps.pitch_data", lambda *a, **k: empty_outing_df)

    resp = client.get("/pitch-types", params={"pitcher_name": "Paul Skenes", "date": "2024-08-05"})

    assert resp.status_code == 200
    assert resp.json() == []


def test_pitch_types_caches_derived_result(monkeypatch, outing_df, client):
    calls = {"n": 0}

    def fake_pitch_data(*a, **k):
        calls["n"] += 1
        return outing_df

    monkeypatch.setattr("pitchviz.api.deps.pitch_data", fake_pitch_data)

    client.get("/pitch-types", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04"})
    client.get("/pitch-types", params={"pitcher_name": "Paul Skenes", "date": "2024-08-04"})

    assert calls["n"] == 1


def test_pitch_types_fetch_failure_returns_500(monkeypatch, client):
    def boom(*a, **k):
        raise RuntimeError("Failed to retrieve pitch data")

    monkeypatch.setattr("pitchviz.api.deps.pitch_data", boom)

    resp = client.get("/pitch-types", params={"pitcher_name": "Nobody", "date": "2024-08-04"})

    assert resp.status_code == 500
