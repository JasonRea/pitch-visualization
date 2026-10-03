import pandas as pd

from pitchviz.data.filters import EVENT_MAP


def _opt(value):
    return None if pd.isna(value) else value


def at_bat_sequences(df: pd.DataFrame) -> list[dict]:
    """Group one outing's pitches by at-bat, in chronological order.

    pybaseball.statcast_pitcher() returns rows most-recent-at-bat-first, so
    this explicitly re-sorts rather than trusting row order.
    """
    rows = df.sort_values(["at_bat_number", "pitch_number"])

    at_bats = []
    for ab_number, group in rows.groupby("at_bat_number", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]

        pitches = []
        for _, row in group.iterrows():
            pitches.append({
                "pitch_number":  int(row["pitch_number"]),
                "pitch_type":    row["pitch_type"],
                "pitch_name":    _opt(row.get("pitch_name")),
                "release_speed": _opt(row.get("release_speed")),
                "balls":         int(row["balls"]),
                "strikes":       int(row["strikes"]),
                "description":   row["description"],
                "type":          row["type"],
            })

        raw_event = last.get("events")
        final_outcome = EVENT_MAP.get(raw_event, raw_event) if pd.notna(raw_event) else None

        at_bats.append({
            "at_bat_number": int(ab_number),
            "inning":        int(first["inning"]),
            "inning_topbot": first["inning_topbot"],
            "stand":         first["stand"],
            "batter":        int(first["batter"]),
            "final_outcome": final_outcome,
            "pitches":       pitches,
        })

    return at_bats
