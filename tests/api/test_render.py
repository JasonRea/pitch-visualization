import respx
from httpx import Response

DISPATCH_URL = "https://api.github.com/repos/test-owner/test-repo/actions/workflows/render_trajectory.yml/dispatches"
RUNS_URL = "https://api.github.com/repos/test-owner/test-repo/actions/runs"
RUN_DETAIL_URL = "https://api.github.com/repos/test-owner/test-repo/actions/runs/777"
RUN_JOBS_URL = "https://api.github.com/repos/test-owner/test-repo/actions/runs/777/jobs"


async def _fast_sleep(_seconds):
    return None


def _request_payload(**overrides):
    payload = {
        "pitcher_name": "Paul Skenes",
        "date": "2024-08-04",
        "split": "all",
        "pitch_type": "",
        "quality": "low_quality",
    }
    payload.update(overrides)
    return payload


@respx.mock
def test_trigger_render_success(monkeypatch, client):
    monkeypatch.setattr("asyncio.sleep", _fast_sleep)
    respx.post(DISPATCH_URL).mock(return_value=Response(204))
    respx.get(RUNS_URL).mock(return_value=Response(200, json={"workflow_runs": [{"id": 777}]}))

    resp = client.post("/render", json=_request_payload())

    assert resp.status_code == 200
    assert resp.json() == {"run_id": 777}


@respx.mock
def test_trigger_render_dispatch_failure_returns_502(monkeypatch, client):
    monkeypatch.setattr("asyncio.sleep", _fast_sleep)
    respx.post(DISPATCH_URL).mock(return_value=Response(422, text="bad ref"))

    resp = client.post("/render", json=_request_payload())

    assert resp.status_code == 502


@respx.mock
def test_trigger_render_timeout_returns_504(monkeypatch, client):
    monkeypatch.setattr("asyncio.sleep", _fast_sleep)
    respx.post(DISPATCH_URL).mock(return_value=Response(204))
    respx.get(RUNS_URL).mock(return_value=Response(200, json={"workflow_runs": []}))

    resp = client.post("/render", json=_request_payload())

    assert resp.status_code == 504


@respx.mock
def test_get_render_status_in_progress(client):
    respx.get(RUN_DETAIL_URL).mock(
        return_value=Response(200, json={"status": "in_progress", "conclusion": None})
    )
    respx.get(RUN_JOBS_URL).mock(
        return_value=Response(200, json={"jobs": [{"steps": [
            {"name": "Set up job", "status": "completed", "conclusion": "success"},
            {"name": "Render pitch", "status": "in_progress", "conclusion": None},
        ]}]})
    )

    resp = client.get("/render/777")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "in_progress"
    assert body["output_url"] is None
    assert body["steps"] == [{"name": "Render pitch", "status": "running"}]


@respx.mock
def test_get_render_status_completed_builds_deterministic_output_url(monkeypatch, client):
    from pitchviz.api.state import render_jobs
    render_jobs[777] = {
        "pitcher_name": "Paul Skenes",
        "date": "2024-08-04",
        "split": "all",
        "pitch_type": "all",
    }
    respx.get(RUN_DETAIL_URL).mock(
        return_value=Response(200, json={"status": "completed", "conclusion": "success"})
    )
    respx.get(RUN_JOBS_URL).mock(return_value=Response(200, json={"jobs": []}))

    resp = client.get("/render/777")

    body = resp.json()
    assert body["status"] == "completed"
    assert body["output_url"] == (
        "https://test-bucket.nyc3.digitaloceanspaces.com"
        "/vizualizations/2024-08-04/paul_skenes-all-all.mp4"
    )


@respx.mock
def test_get_render_status_completed_failure_has_no_output_url(client):
    respx.get(RUN_DETAIL_URL).mock(
        return_value=Response(200, json={"status": "completed", "conclusion": "failure"})
    )
    respx.get(RUN_JOBS_URL).mock(return_value=Response(200, json={"jobs": []}))

    resp = client.get("/render/777")

    body = resp.json()
    assert body["status"] == "failed"
    assert body["output_url"] is None


@respx.mock
def test_get_render_status_run_not_found(client):
    respx.get(RUN_DETAIL_URL).mock(return_value=Response(404))
    respx.get(RUN_JOBS_URL).mock(return_value=Response(404))

    resp = client.get("/render/777")

    assert resp.status_code == 404
