from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def outing_df() -> pd.DataFrame:
    """A real, frozen Statcast outing (Paul Skenes, 2024-08-04 — 100 pitches, 22 at-bats)."""
    return pd.read_csv(FIXTURES_DIR / "outing_2024-08-04.csv")


@pytest.fixture
def empty_outing_df(outing_df) -> pd.DataFrame:
    """Same columns as outing_df, zero rows — the 'pitcher didn't pitch that day' case."""
    return outing_df.iloc[0:0].copy()


@pytest.fixture(autouse=True)
def _clear_api_caches():
    """Module-level TTLCache/dict singletons in api/state.py persist across the whole
    pytest session — clear them before every test so one test's cached result can't
    leak into another."""
    from pitchviz.api import state
    state.players_cache.clear()
    state.pitch_type_cache.clear()
    state.outing_cache.clear()
    yield
    state.players_cache.clear()
    state.pitch_type_cache.clear()
    state.outing_cache.clear()


@pytest.fixture
def client() -> TestClient:
    from pitchviz.api.main import app
    return TestClient(app)
