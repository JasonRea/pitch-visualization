import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pitchviz.api.routers import dates, pitch_types, pitchers, movement, heatmap, at_bats

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")

app = FastAPI(title="Pitch Viz API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(pitchers.router)
app.include_router(dates.router)
app.include_router(pitch_types.router)
app.include_router(movement.router)
app.include_router(heatmap.router)
app.include_router(at_bats.router)
