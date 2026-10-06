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

PITCH_DESCRIPTION_MAP = {
    "ball": "Ball",
    "called_strike": "Called Strike",
    "swinging_strike": "Swinging Strike",
    "swinging_strike_blocked": "Swinging Strike",
    "foul": "Foul",
    "hit_into_play": "In Play",
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

def pitches_filter_by_at_bat(at_bat_number: int, label: str | None = None):
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[["vx0", "vy0", "vz0", "ax", "ay", "az",
                "release_pos_x", "release_pos_z", "release_pos_y",
                "pitch_type", "pitch_name", "release_speed", "description", "events",
                "at_bat_number", "pitch_number", "batter"]]
            # events is NaN by design on every pitch but the one that ends
            # the at-bat — a blanket dropna() would drop all the others.
            .dropna(subset=["vx0", "vy0", "vz0", "ax", "ay", "az",
                             "release_pos_x", "release_pos_z", "release_pos_y", "pitch_type"])
            .loc[lambda d: d["at_bat_number"] == at_bat_number]
            .sort_values("pitch_number")
        )
    _filter._label = label or f"At-Bat #{at_bat_number}"
    return _filter


def pitches_filter_by_pitch_numbers(at_bat_number: int, pitch_numbers: list[int]):
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[["vx0", "vy0", "vz0", "ax", "ay", "az",
                "release_pos_x", "release_pos_z", "release_pos_y",
                "pitch_type", "pitch_name", "release_speed", "description", "events",
                "at_bat_number", "pitch_number", "batter"]]
            .dropna(subset=["vx0", "vy0", "vz0", "ax", "ay", "az",
                             "release_pos_x", "release_pos_z", "release_pos_y", "pitch_type"])
            .loc[lambda d: (d["at_bat_number"] == at_bat_number) & (d["pitch_number"].isin(pitch_numbers))]
            .sort_values("pitch_number")
        )
    _filter._label = f"Tunnel: pitches {', '.join(str(n) for n in sorted(pitch_numbers))} (AB #{at_bat_number})"
    return _filter


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
