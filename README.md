# Pitch Viz

A suite of automated pitch visualization graphics and animations, backed by a FastAPI API and a Manim/pybaseball rendering pipeline.

## Project layout

```
src/pitchviz/
├── config.py       # shared, environment-independent constants (pitch colors/names)
├── api/            # FastAPI app
│   ├── main.py     # app factory, CORS, router registration
│   ├── state.py    # in-process caches
│   └── routers/    # one module per resource (pitchers, dates, pitch_types, movement, heatmap, at_bats)
├── data/           # Statcast/MLB data fetching and dataframe filters
│   ├── fetch.py
│   └── filters.py
└── render/         # Manim scene building and the render CLI
    ├── builder.py
    ├── assets.py   # player bio lookups
    └── run.py
```

## Installing

```bash
pip install -e .            # API-only dependencies
pip install -e ".[render]"  # + manim for local rendering
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/pitchers?q={name}` | Search active MLB pitchers by name (min 2 chars) |
| `GET` | `/dates?pitcher_name={name}&season={year}` | Get dates the pitcher appeared in games |
| `GET` | `/pitch-types?pitcher_name={name}&date={YYYY-MM-DD}` | Get pitch types thrown in a specific outing |
| `GET` | `/movement?pitcher_name={name}&date={YYYY-MM-DD}` | Per-pitch and summary movement stats (velo, horizontal/induced vertical break) for an outing |
| `GET` | `/heatmap?pitcher_name={name}&date={YYYY-MM-DD}&pitch_type={code}&stand={L\|R}` | Pitch locations for a strike-zone heatmap, optionally filtered by pitch type and/or batter side |
| `GET` | `/at-bats?pitcher_name={name}&date={YYYY-MM-DD}` | Full at-bat-by-at-bat pitch sequences for an outing, with final outcomes |

`/pitch-types`, `/movement`, `/heatmap`, and `/at-bats` all share a 1-hour in-process cache keyed on `pitcher_name:date`, so hitting more than one of them for the same outing only fetches Statcast data once.

Interactive docs available at `/docs` when the server is running.

## Running the API locally

```bash
pip install -e .
uvicorn pitchviz.api.main:app --reload
```

API is available at `http://localhost:8000`. No environment variables are required; `ALLOWED_ORIGIN` optionally restricts CORS (defaults to `*`).

## Rendering pitches from the CLI

```bash
pip install -e ".[render]"
python -m pitchviz.render.run -d "2026-02-24" "Ranger Suarez" "high_quality"
```

Run `python -m pitchviz.render.run` with no arguments to see all available options (single pitch type, splits, per-pitcher daily renders, etc.). Rendering runs entirely locally via Manim and writes output to disk — no external services involved.

## Caveats

- **`/pitch-types` latency** — calls Statcast via pybaseball, takes 3–8 seconds. Results are cached for 1 hour per pitcher+date pair.

## Deployment

Not yet set up — this project currently runs locally only. Hosting and automated/remote rendering are deliberately out of scope for now and will be revisited as a separate decision.
