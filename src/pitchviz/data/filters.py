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

def high_heat_filter(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_keep = [
            'pitcher',
            'pitch_type',
            'pitch_name',
            'release_spin_rate',
            'pfx_x', 'pfx_z', # HB is represented by pfx_x and iVB is represented by pfx_z
            #'estimated_woba_using_speedangle', NOTE Let's bring this back later, not great with spring training data I guess
            'release_speed'
        ]
    return df[columns_to_keep].dropna().sort_values('release_speed', ascending=False).head(5)

def absolute_missiles_filter(df: pd.DataFrame) -> pd.DataFrame:

    columns_to_keep = ['launch_speed',
                   'launch_angle',
                   'batter',
                   'events',
                   'bb_type',
                   'hit_distance_sc',
                   'estimated_ba_using_speedangle',
                   ]

    df = df[columns_to_keep]

    df = df[df['bb_type'] == 'fly_ball']

    df = df.drop('bb_type', axis=1)

    df['events'] = df['events'].map(EVENT_MAP)

    return df.sort_values('launch_speed', ascending=False).head(5)

def big_five_filter(df: pd.DataFrame) -> pd.DataFrame:

    columns_to_keep = [
                   'batter',
                   'pitcher',
                   'delta_home_win_exp',
                   'delta_run_exp',
                   "events",
                   "inning",
                   'des',
                   ]

    df = df[columns_to_keep]

    df['events'] = df['events'].map(EVENT_MAP)

    return df.sort_values('delta_home_win_exp', ascending=False).head(5)

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
