import numpy as np
import pandas as pd
import pytest

from pitchviz.data.filters import pitches_filter, pitches_filter_by_at_bat
from pitchviz.render.builder import (
    position,
    abs_strike_zone,
    pitch_annotation_text,
    box_score_line,
    pitch_type_breakdown,
    VizualizationBuilder,
)


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


def test_pitch_annotation_text_mid_at_bat_has_no_outcome_line():
    # Real values: at-bat 1, pitch 1 of the fixture outing (not the final pitch).
    text = pitch_annotation_text({
        "release_speed": 95.2,
        "pitch_name": "Split-Finger",
        "description": "called_strike",
        "events": None,
    })

    assert text == "Split-Finger · 95.2 mph\nCalled Strike"
    assert "Outcome" not in text


def test_pitch_annotation_text_final_pitch_includes_outcome_line():
    # Real values: at-bat 1, pitch 3 of the fixture outing (ends the at-bat).
    text = pitch_annotation_text({
        "release_speed": 94.3,
        "pitch_name": "Split-Finger",
        "description": "hit_into_play",
        "events": "single",
    })

    assert text == "Split-Finger · 94.3 mph\nIn Play\nOutcome: Single"


def test_box_score_line_formats_all_six_stats():
    line = box_score_line({
        "innings_pitched": "6.1", "strikeouts": 10, "walks": 1,
        "hits": 2, "runs": 0, "earned_runs": 0,
    })

    assert line == "6.1 IP  10 K  1 BB  2 H  0 R  0 ER"


def test_pitch_type_breakdown_computes_rate_stats_from_real_data(outing_df):
    filtered = pitches_filter(outing_df)

    rows = {r["pitch_type"]: r for r in pitch_type_breakdown(filtered)}
    ff = rows["FF"]

    # Hand-computed from the real fixture: 40 FF pitches, 15 in-zone (of 40
    # with a known zone), 25 out-of-zone with 5 chased, 11 swings with 2 whiffs.
    assert ff["count"] == 40
    assert ff["avg_ivb"] == pytest.approx(9.639, abs=0.01)
    assert ff["avg_hvb"] == pytest.approx(-14.214, abs=0.01)
    assert ff["zone_pct"] == pytest.approx(37.5, abs=0.01)
    assert ff["chase_pct"] == pytest.approx(20.0, abs=0.01)
    assert ff["whiff_pct"] == pytest.approx(18.18, abs=0.01)


def test_pitch_type_breakdown_handles_pitch_type_with_no_out_of_zone_pitches():
    df = pd.DataFrame({
        "pitch_type": ["FF", "FF"],
        "release_speed": [97.0, 98.0],
        "release_spin_rate": [2300, 2310],
        "pfx_x": [-1.0, -1.1],
        "pfx_z": [0.8, 0.9],
        "zone": [1, 2],  # both in-zone — no out-of-zone denominator
        "description": ["called_strike", "ball"],
    })

    rows = pitch_type_breakdown(df)

    assert rows[0]["chase_pct"] is None
    assert rows[0]["whiff_pct"] is None  # no swings either


def test_resolve_game_summary_computes_pitch_type_breakdown_from_real_data(monkeypatch, outing_df):
    filtered = pitches_filter(outing_df)
    raw_df = outing_df.copy()
    raw_df["pitcher"] = 694973
    raw_df["game_date"] = "2024-08-04"
    raw_df["game_pk"] = 745468

    monkeypatch.setattr(
        "pitchviz.render.builder.get_pitcher_game_stats",
        lambda player_id, season, game_pk: {
            "innings_pitched": "6.1", "strikeouts": 11, "walks": 2,
            "hits": 3, "runs": 1, "earned_runs": 1,
        },
    )
    monkeypatch.setattr(
        "pitchviz.render.builder.get_player_names",
        lambda ids: {694973: "Paul Skenes"},
    )
    fake_headshot = np.zeros((4, 4, 4), dtype=np.uint8)
    monkeypatch.setattr(
        "pitchviz.render.builder.get_player_headshot",
        lambda player_id: fake_headshot,
    )

    builder = VizualizationBuilder()
    builder._resolve_game_summary(raw_df, filtered, pitches_filter)

    assert builder._game_summary_data is not None
    assert builder._game_summary_data["box_score"] == {
        "innings_pitched": "6.1", "strikeouts": 11, "walks": 2,
        "hits": 3, "runs": 1, "earned_runs": 1,
    }
    assert builder._game_summary_data["pitcher_name"] == "Paul Skenes"
    assert builder._game_summary_data["date"] == "2024-08-04"
    assert builder._game_summary_data["headshot"] is fake_headshot
    rows = {r["pitch_type"]: r for r in builder._game_summary_data["pitch_type_rows"]}
    assert rows["FF"]["count"] == 40  # FF is the most-thrown pitch type (40 of 100)
    assert rows["FF"]["name"] == "4-Seam Fastball"
    assert rows["FF"]["avg_velo"] == pytest.approx(98.04, abs=0.01)


def test_resolve_game_summary_skips_non_full_outing_scopes(outing_df):
    filt = pitches_filter_by_at_bat(1)
    filtered = filt(outing_df)

    builder = VizualizationBuilder()
    builder._resolve_game_summary(outing_df, filtered, filt)

    assert builder._game_summary_data is None


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
