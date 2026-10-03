from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pitchviz.api.config import ALLOWED_ORIGIN
from pitchviz.api.routers import dates, pitch_types, pitchers, render

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
app.include_router(render.router)
