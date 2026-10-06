from pitchviz.data.filters import (
    EVENT_MAP,
    PITCH_DESCRIPTION_MAP,
    pitches_filter,
    pitches_filter_vs_left,
    pitches_filter_vs_right,
    pitches_filter_by_pitch_type,
    pitches_filter_by_at_bat,
    pitches_filter_by_pitch_numbers,
    location_filter,
)


def test_pitches_filter_drops_rows_missing_kinematics(outing_df):
    result = pitches_filter(outing_df)

    assert not result.empty
    assert result[["vx0", "vy0", "vz0", "ax", "ay", "az"]].isna().sum().sum() == 0


def test_pitches_filter_vs_left_only_keeps_left_batters(outing_df):
    result = pitches_filter_vs_left(outing_df)

    assert not result.empty
    assert (result["stand"] == "L").all()


def test_pitches_filter_vs_right_only_keeps_right_batters(outing_df):
    result = pitches_filter_vs_right(outing_df)

    assert (result["stand"] == "R").all()


def test_pitches_filter_by_pitch_type_filters_and_tags_itself(outing_df):
    filt = pitches_filter_by_pitch_type("SL")
    result = filt(outing_df)

    assert filt._pitch_type == "SL"
    assert (result["pitch_type"] == "SL").all()
    assert len(result) == 4


def test_location_filter_no_args_keeps_all_rows_with_plate_coords(outing_df):
    result = location_filter(outing_df)

    assert len(result) == 100
    assert result[["plate_x", "plate_z"]].isna().sum().sum() == 0


def test_location_filter_by_stand(outing_df):
    result = location_filter(outing_df, stand="L")

    assert len(result) == 88
    assert (result["stand"] == "L").all()


def test_location_filter_by_pitch_type(outing_df):
    result = location_filter(outing_df, pitch_type="SL")

    assert len(result) == 4
    assert (result["pitch_type"] == "SL").all()


def test_location_filter_combines_both_filters(outing_df):
    result = location_filter(outing_df, pitch_type="FF", stand="L")

    assert (result["pitch_type"] == "FF").all()
    assert (result["stand"] == "L").all()


def test_pitches_filter_by_at_bat_filters_and_sorts_by_pitch_number(outing_df):
    filt = pitches_filter_by_at_bat(1)
    result = filt(outing_df)

    assert filt._label == "At-Bat #1"
    assert (result["at_bat_number"] == 1).all()
    assert list(result["pitch_number"]) == [1, 2, 3]


def test_pitches_filter_by_at_bat_keeps_batter_column(outing_df):
    filt = pitches_filter_by_at_bat(1)
    result = filt(outing_df)

    assert "batter" in result.columns
    assert result["batter"].nunique() == 1


def test_pitches_filter_by_at_bat_keeps_non_final_pitches_despite_null_events(outing_df):
    # Regression: events is NaN by design on every pitch but the one that
    # ends the at-bat — a blanket dropna() would wrongly drop the others.
    filt = pitches_filter_by_at_bat(1)
    result = filt(outing_df)

    assert "release_speed" in result.columns
    assert "description" in result.columns
    assert "events" in result.columns
    assert len(result) == 3  # at-bat 1 is a 3-pitch walk in the fixture
    assert result["events"].isna().sum() == 2
    assert result["events"].notna().sum() == 1


def test_pitches_filter_by_at_bat_accepts_custom_label(outing_df):
    filt = pitches_filter_by_at_bat(1, label="vs Corbin Carroll — Inning 1")

    assert filt._label == "vs Corbin Carroll — Inning 1"


def test_pitches_filter_by_pitch_numbers_filters_to_selected_pitches(outing_df):
    filt = pitches_filter_by_pitch_numbers(1, [1, 3])
    result = filt(outing_df)

    assert "Tunnel" in filt._label
    assert (result["at_bat_number"] == 1).all()
    assert list(result["pitch_number"]) == [1, 3]
    assert "batter" in result.columns
    assert result["batter"].nunique() == 1
    assert "release_speed" in result.columns
    assert "description" in result.columns
    assert result["events"].isna().tolist() == [True, False]  # pitch 1: no outcome, pitch 3: the walk


def test_pitch_description_map_covers_real_statcast_descriptions():
    for raw in ["ball", "called_strike", "hit_into_play", "foul", "swinging_strike", "swinging_strike_blocked"]:
        assert raw in PITCH_DESCRIPTION_MAP


def test_event_map_covers_common_outcomes():
    assert EVENT_MAP["single"] == "Single"
    assert EVENT_MAP["strikeout"] == "Strikeout"
    assert EVENT_MAP["grounded_into_double_play"] == "GIDP"
