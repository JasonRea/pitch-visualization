import pandas as pd

EVENT_MAP = {
    "single" : "Single",
    "double" : "Double",
    "triple" : "Triple",
    "home_run" : "Home Run",
    "field_out" : "Flyout",
    "strikeout" : "Strikeout",
    "grounded_into_double_play" : "GIDP",
    "force_out" : "Force Out",
    "fielders_choice" : "Fielder's Choice",
    "sac_bunt" : "Sac Bunt",
    "hit_by_pitch" : "Hit By Pitch",
    "walk" : "Walk",
    "sac_fly" : "Sac Fly",
    "double_play" : "Double Play",
    "field_error" : "Error",
    "fielders_choice_out" : "Fielder's Choice"

}

# -----FILTERS----------

def pitches_filter(df: pd.DataFrame):
    columns_to_keep = [
            "vx0", "vy0", "vz0",
            "ax", "ay", "az",
            "release_pos_x", "release_pos_z", "release_pos_y",
            "pitch_type",
        ]

    return df[columns_to_keep].dropna()

def pitches_filter_vs_left(df: pd.DataFrame):
    columns_to_keep = [
            "vx0", "vy0", "vz0",
            "ax", "ay", "az",
            "release_pos_x", "release_pos_z", "release_pos_y",
            "pitch_type", 'stand'
        ]

    df = df[columns_to_keep].dropna()

    return df[df['stand'] == 'L']

def pitches_filter_vs_right(df: pd.DataFrame):
    columns_to_keep = [
            "vx0", "vy0", "vz0",
            "ax", "ay", "az",
            "release_pos_x", "release_pos_z", "release_pos_y",
            "pitch_type", 'stand'
        ]

    df = df[columns_to_keep].dropna()

    return df[df['stand'] == 'R']

def pitches_filter_by_pitch_type(pitch_type_code: str):
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[["vx0", "vy0", "vz0", "ax", "ay", "az",
                "release_pos_x", "release_pos_z", "release_pos_y",
                "pitch_type", "pitch_name"]]
            .dropna()
            .loc[lambda d: d["pitch_type"] == pitch_type_code]
        )
    _filter._pitch_type = pitch_type_code
    return _filter

# TODO list valid names
def pitches_filter_by_name(df: pd.DataFrame, pitch_name: str):
    columns_to_keep = [
            "vx0", "vy0", "vz0",
            "ax", "ay", "az",
            "release_pos_x", "release_pos_z", "release_pos_y",
            "pitch_type", 'stand'
        ]

    df = df[columns_to_keep].dropna()

    return df[df['pitch_type'] == pitch_name]

def location_filter(df: pd.DataFrame, pitch_type: str | None = None, stand: str | None = None) -> pd.DataFrame:
    columns_to_keep = [
            "pitch_type", "plate_x", "plate_z", "zone",
            "sz_top", "sz_bot", "stand", "description",
        ]

    df = df[columns_to_keep].dropna(subset=["plate_x", "plate_z"])

    if pitch_type:
        df = df[df["pitch_type"] == pitch_type]
    if stand:
        df = df[df["stand"] == stand]

    return df
