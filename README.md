# Pitch Viz

A suite of automated pitch visualization graphics and animations, backed by a FastAPI API and a Manim/pybaseball rendering pipeline.

## Project layout

```
src/pitchviz/
├── config.py       # shared, environment-independent constants (pitch colors/names)
├── api/            # FastAPI app
│   ├── main.py     # app factory, CORS, router registration
│   ├── config.py   # API-only env vars (GitHub, DO Spaces, CORS)
│   ├── state.py    # in-process caches and render job store
│   └── routers/    # one module per resource (pitchers, dates, pitch_types, render)
├── data/           # Statcast/MLB data fetching and dataframe filters
│   ├── fetch.py
│   └── filters.py
└── render/         # Manim scene building and the render CLI
    ├── builder.py
    ├── assets.py   # headshots/bios/team logos (Pillow-dependent)
    └── run.py
```

## Installing

```bash
pip install -e .            # API-only dependencies
pip install -e ".[render]"  # + manim/matplotlib/boto3/PyGithub for rendering
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/pitchers?q={name}` | Search active MLB pitchers by name (min 2 chars) |
| `GET` | `/dates?pitcher_name={name}&season={year}` | Get dates the pitcher appeared in games |
| `GET` | `/pitch-types?pitcher_name={name}&date={YYYY-MM-DD}` | Get pitch types thrown in a specific outing |
| `POST` | `/render` | Trigger a GitHub Actions render job |
| `GET` | `/render/{run_id}` | Poll render job status |

Interactive docs available at `/docs` when the server is running.

### POST /render — request body

```json
{
  "pitcher_name": "Ranger Suarez",
  "date": "2026-04-18",
  "split": "all",
  "pitch_type": "SL",
  "quality": "low_quality"
}
```

- `split`: `"all"` | `"left"` | `"right"`
- `pitch_type`: 2-letter Statcast code (e.g. `"FF"`, `"SL"`) or `""` for all pitch types
- `quality`: `"low_quality"` | `"medium_quality"` | `"high_quality"` | `"fourk_quality"`

Returns `{ "run_id": 12345 }`. Poll `/render/{run_id}` every 10s until `status` is `"completed"` or `"failed"`. On success, `output_url` contains the public MP4 link.

## Running the API locally

```bash
pip install -e .
```

Set environment variables (see `.env.example`):

```bash
export GITHUB_TOKEN=ghp_...          # Fine-grained PAT with Actions: write
export GITHUB_OWNER=your_username
export GITHUB_REPO=pitch-vizualization
export DO_SPACES_REGION=nyc3
export DO_SPACES_BUCKET=your_bucket
export ALLOWED_ORIGIN=http://localhost:5173
```

Start the server:

```bash
uvicorn pitchviz.api.main:app --reload
```

API is available at `http://localhost:8000`.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GITHUB_TOKEN` | Yes | GitHub PAT with `Actions: write` permission |
| `GITHUB_OWNER` | Yes | GitHub username or org that owns the repo |
| `GITHUB_REPO` | Yes | Repository name (e.g. `pitch-vizualization`) |
| `DO_SPACES_REGION` | Yes | DigitalOcean Spaces region (e.g. `nyc3`) |
| `DO_SPACES_BUCKET` | Yes | Spaces bucket name |
| `ALLOWED_ORIGIN` | No | CORS-allowed origin for API clients (defaults to `*`) |

## Deploying to Render.com

A `render.yaml` at the repo root can auto-configure a Render.com web service (build command `pip install -e .`, start command `uvicorn pitchviz.api.main:app --host 0.0.0.0 --port $PORT`). Set the 6 env vars above in the Render dashboard (Dashboard → your service → Environment).

The free tier spins down after 15 minutes of inactivity. The first request after idle takes ~30 seconds to cold-start.

## Rendering pitches from the CLI

```bash
pip install -e ".[render]"
python -m pitchviz.render.run -d "2026-02-24" "Ranger Suarez" "high_quality"
```

Run `python -m pitchviz.render.run` with no arguments to see all available options (single pitch type, splits, daily graphics, etc.).

## How rendering works

`POST /render` dispatches the `.github/workflows/render_trajectory.yml` workflow via the GitHub API. It then polls for up to 12 seconds to find the new run ID and returns it. The run input is stored in memory so the output URL can be reconstructed when the job finishes.

Output MP4s are uploaded to DigitalOcean Spaces at:
```
vizualizations/trajectories/{date}/{pitcher_slug}-{split}-{pitch_type}.mp4
```

The URL is deterministic from the inputs, so no callback from Actions back to the API is needed.

## Caveats

- **Render.com cold starts** — first request after 15 min idle is slow (~30s).
- **In-memory job store** — `render_jobs` (in `pitchviz.api.state`) is lost on server restart. If the server restarts between `/render` and `/render/{run_id}`, the status endpoint won't be able to construct `output_url` (it will return `null` even on success).
- **`/pitch-types` latency** — calls Statcast via pybaseball, takes 3–8 seconds. Results are cached for 1 hour per pitcher+date pair.
- **GitHub Actions PAT** — the token needs `Actions: write` on the target repo. A fine-grained PAT scoped to just that repo is recommended over a classic PAT.
