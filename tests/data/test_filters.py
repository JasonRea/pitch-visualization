from pitchviz.data.filters import (
    EVENT_MAP,
    pitches_filter,
    pitches_filter_vs_left,
    pitches_filter_vs_right,
    pitches_filter_by_pitch_type,
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


def test_event_map_covers_common_outcomes():
    assert EVENT_MAP["single"] == "Single"
    assert EVENT_MAP["strikeout"] == "Strikeout"
    assert EVENT_MAP["grounded_into_double_play"] == "GIDP"
