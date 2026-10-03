import pytest

from pitchviz.data.stats import movement_rows, movement_summary


def test_movement_rows_count_and_ordering(outing_df):
    rows = movement_rows(outing_df)

    assert len(rows) == 100
    # chronological: (at_bat_number, pitch_number) non-decreasing
    keys = [(r["at_bat_number"], r["pitch_number"]) for r in rows]
    assert keys == sorted(keys)


def test_movement_rows_converts_feet_to_inches(outing_df):
    rows = movement_rows(outing_df)
    first = rows[0]

    assert first["pitch_type"] == "FS"
    assert first["release_speed"] == 95.2
    assert first["pfx_x_in"] == pytest.approx(-15.24)
    assert first["pfx_z_in"] == pytest.approx(-0.6)
    assert first["release_spin_rate"] == 1786.0
    assert first["spin_axis"] == 252.0


def test_movement_summary_golden_values(outing_df):
    summary = {s["code"]: s for s in movement_summary(outing_df)}

    assert set(summary) == {"FF", "FS", "CU", "CH", "SL", "ST"}

    ff = summary["FF"]
    assert ff["name"] == "4-Seam Fastball"
    assert ff["color"] == "#FF007D"
    assert ff["count"] == 40
    assert ff["avg_velo"] == pytest.approx(98.0425)
    assert ff["avg_hb_in"] == pytest.approx(-14.214)
    assert ff["avg_ivb_in"] == pytest.approx(9.639)
    assert ff["avg_spin_rate"] == pytest.approx(2235.975)


def test_movement_summary_counts_sum_to_total_pitches(outing_df):
    summary = movement_summary(outing_df)

    assert sum(s["count"] for s in summary) == len(outing_df)


def test_movement_summary_sorted_by_count_desc(outing_df):
    summary = movement_summary(outing_df)

    counts = [s["count"] for s in summary]
    assert counts == sorted(counts, reverse=True)


def test_movement_on_empty_outing_returns_empty(empty_outing_df):
    assert movement_rows(empty_outing_df) == []
    assert movement_summary(empty_outing_df) == []
