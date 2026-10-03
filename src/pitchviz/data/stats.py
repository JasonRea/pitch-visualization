import pandas as pd

from pitchviz.config import PITCH_COLORS, PITCH_NAMES


def _opt_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def movement_rows(df: pd.DataFrame) -> list[dict]:
    """Per-pitch movement data for one outing, in release/at-bat order."""
    required = ["release_speed", "pfx_x", "pfx_z"]
    rows = df.dropna(subset=required).sort_values(["at_bat_number", "pitch_number"])

    result = []
    for _, row in rows.iterrows():
        result.append({
            "at_bat_number":     int(row["at_bat_number"]),
            "pitch_number":      int(row["pitch_number"]),
            "pitch_type":        row["pitch_type"],
            "pitch_name":        row.get("pitch_name") or PITCH_NAMES.get(row["pitch_type"], row["pitch_type"]),
            "release_speed":     float(row["release_speed"]),
            "pfx_x_in":          float(row["pfx_x"]) * 12,
            "pfx_z_in":          float(row["pfx_z"]) * 12,
            "release_spin_rate": _opt_float(row.get("release_spin_rate")),
            "spin_axis":         _opt_float(row.get("spin_axis")),
        })
    return result


def movement_summary(df: pd.DataFrame) -> list[dict]:
    """Per-pitch-type movement averages for one outing."""
    required = ["release_speed", "pfx_x", "pfx_z"]
    rows = df.dropna(subset=required)

    result = []
    for code, group in rows.groupby("pitch_type"):
        name = group["pitch_name"].dropna().iloc[0] if group["pitch_name"].notna().any() else PITCH_NAMES.get(code, code)
        result.append({
            "code":            code,
            "name":            name,
            "color":           PITCH_COLORS.get(code, "#9C8975"),
            "count":           int(len(group)),
            "avg_velo":        float(group["release_speed"].mean()),
            "avg_hb_in":       float(group["pfx_x"].mean()) * 12,
            "avg_ivb_in":      float(group["pfx_z"].mean()) * 12,
            "avg_spin_rate":   _opt_float(group["release_spin_rate"].mean()) if "release_spin_rate" in group else None,
            "avg_spin_axis":   _opt_float(group["spin_axis"].mean()) if "spin_axis" in group else None,
        })

    result.sort(key=lambda x: -x["count"])
    return result
