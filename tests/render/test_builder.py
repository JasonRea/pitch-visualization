import numpy as np
import pytest

from pitchviz.data.filters import pitches_filter, pitches_filter_by_at_bat
from pitchviz.render.builder import position, abs_strike_zone, VizualizationBuilder


def test_position_at_t_zero_is_release_point():
    p = position(0, x0=1.0, y0=60.5, z0=5.5, vx0=0, vy0=-130, vz0=-5, ax=0, ay=20, az=-16)

    assert p == pytest.approx(np.array([1.0, 60.5, 5.5]))


def test_position_matches_kinematic_equation_at_t():
    x0, y0, z0 = 0.0, 60.5, 5.5
    vx0, vy0, vz0 = 2.0, -130.0, -4.0
    ax, ay, az = -10.0, 25.0, -18.0
    t = 0.4

    p = position(t, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)

    expected = np.array([
        x0 + vx0 * t + 0.5 * ax * t**2,
        y0 + vy0 * t + 0.5 * ay * t**2,
        z0 + vz0 * t + 0.5 * az * t**2,
    ])
    assert p == pytest.approx(expected)


def test_position_y_decreases_toward_plate_over_time():
    # A pitch travels from the mound (y~60.5) toward home plate (y=0): y should fall.
    p0 = position(0.0,  x0=0, y0=60.5, z0=5.5, vx0=0, vy0=-130, vz0=0, ax=0, ay=20, az=0)
    p1 = position(0.3,  x0=0, y0=60.5, z0=5.5, vx0=0, vy0=-130, vz0=0, ax=0, ay=20, az=0)

    assert p1[1] < p0[1]


def test_abs_strike_zone_matches_known_values():
    # A 6' (72in) batter: bottom at 27%, top at 53.5% of height, in feet.
    bottom, top = abs_strike_zone(72)

    assert bottom == pytest.approx(1.62)
    assert top == pytest.approx(3.21)


def test_resolve_strike_zone_uses_batter_height_for_single_batter_df(monkeypatch, outing_df):
    filt = pitches_filter_by_at_bat(1)
    df = filt(outing_df)
    batter_id = int(df["batter"].iloc[0])

    monkeypatch.setattr(
        "pitchviz.render.builder.get_player_heights",
        lambda ids: {batter_id: 72},
    )

    builder = VizualizationBuilder()
    builder._resolve_strike_zone(df)

    assert builder._sz_bottom == pytest.approx(1.62)
    assert builder._sz_top == pytest.approx(3.21)


def test_resolve_strike_zone_keeps_default_for_multi_batter_df(monkeypatch, outing_df):
    df = pitches_filter(outing_df)  # full outing — no "batter" column, many batters

    def _boom(ids):
        raise AssertionError("should not look up heights for a multi-batter df")

    monkeypatch.setattr("pitchviz.render.builder.get_player_heights", _boom)

    builder = VizualizationBuilder()
    default_bottom, default_top = builder._sz_bottom, builder._sz_top
    builder._resolve_strike_zone(df)

    assert builder._sz_bottom == default_bottom
    assert builder._sz_top == default_top
