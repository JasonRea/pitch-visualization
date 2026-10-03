import asyncio
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pitchviz.api.config import GITHUB_OWNER, GITHUB_REPO, GH_HEADERS, DO_SPACES_REGION, DO_SPACES_BUCKET
from pitchviz.api.state import render_jobs

router = APIRouter()


# ---------------------------------------------------------------------------
# POST /render
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    pitcher_name: str
    date:         str
    split:        str = "all"
    pitch_type:   str = ""
    quality:      str = "low_quality"


@router.post("/render")
async def trigger_render(req: RenderRequest):
    before_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
            f"/actions/workflows/render_trajectory.yml/dispatches",
            headers=GH_HEADERS,
            json={
                "ref": "main",
                "inputs": {
                    "pitcher_name": req.pitcher_name,
                    "date":         req.date,
                    "split":        req.split or "all",
                    "pitch_type":   req.pitch_type or "",
                    "quality":      req.quality or "low_quality",
                },
            },
            timeout=15,
        )
    if r.status_code != 204:
        raise HTTPException(502, detail=f"GitHub API error {r.status_code}: {r.text}")

    # Poll up to ~12s for the newly created run to appear
    run_id: int | None = None
    async with httpx.AsyncClient() as client:
        for _ in range(6):
            await asyncio.sleep(2)
            r = await client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs",
                headers=GH_HEADERS,
                params={
                    "workflow_id": "render_trajectory.yml",
                    "event":       "workflow_dispatch",
                    "created":     f">={before_ts}",
                },
                timeout=10,
            )
            runs = r.json().get("workflow_runs", [])
            if runs:
                run_id = runs[0]["id"]
                break

    if run_id is None:
        raise HTTPException(504, detail="Timed out waiting for GitHub Actions run to start")

    render_jobs[run_id] = {
        "pitcher_name": req.pitcher_name,
        "date":         req.date,
        "split":        req.split or "all",
        "pitch_type":   req.pitch_type or "all",
    }

    return {"run_id": run_id}


# ---------------------------------------------------------------------------
# GET /render/{run_id}
# ---------------------------------------------------------------------------

_SKIP_STEPS = {"Set up job", "Complete job"}


def _map_step_status(step: dict) -> str:
    gh_status   = step.get("status", "queued")
    conclusion  = step.get("conclusion")
    if gh_status == "in_progress":
        return "running"
    if gh_status == "completed":
        return "done" if conclusion == "success" else "failed"
    return "pending"


@router.get("/render/{run_id}")
async def get_render_status(run_id: int):
    async with httpx.AsyncClient() as client:
        run_resp, jobs_resp = await asyncio.gather(
            client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}",
                headers=GH_HEADERS,
                timeout=10,
            ),
            client.get(
                f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{run_id}/jobs",
                headers=GH_HEADERS,
                timeout=10,
            ),
        )

    if run_resp.status_code == 404:
        raise HTTPException(404, detail="Run not found")
    run_resp.raise_for_status()

    run        = run_resp.json()
    status     = run["status"]
    conclusion = run.get("conclusion")

    # Extract steps from the first job, filtering out GitHub infrastructure noise
    steps: list[dict] = []
    if jobs_resp.status_code == 200:
        jobs = jobs_resp.json().get("jobs", [])
        if jobs:
            steps = [
                {"name": s["name"], "status": _map_step_status(s)}
                for s in jobs[0].get("steps", [])
                if s.get("name") not in _SKIP_STEPS
            ]

    output_url = None
    if status == "completed" and conclusion == "success":
        job          = render_jobs.get(run_id, {})
        pitcher_slug = job.get("pitcher_name", "").lower().replace(" ", "_")
        date         = job.get("date", "")
        split        = job.get("split", "all")
        pitch_type   = job.get("pitch_type", "all")
        output_url   = (
            f"https://{DO_SPACES_BUCKET}.{DO_SPACES_REGION}.digitaloceanspaces.com"
            f"/vizualizations/{date}/{pitcher_slug}-{split}-{pitch_type}.mp4"
        )

    api_status = {
        "queued":      "queued",
        "in_progress": "in_progress",
        "completed":   "completed" if conclusion == "success" else "failed",
    }.get(status, status)

    return {"status": api_status, "output_url": output_url, "steps": steps}
