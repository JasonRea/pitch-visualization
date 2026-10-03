"""Shared, environment-independent constants used by both the API and the render pipeline."""

# Borrowed from TJstats pitch color palette
PITCH_COLORS = {
    "FF": "#FF007D", "FA": "#FF007D", "SI": "#98165D", "FC": "#BE5FA0",
    "CH": "#F79E70", "FS": "#FE6100", "SC": "#F08223", "FO": "#FFB000",
    "SL": "#67E18D", "ST": "#1BB999", "SV": "#376748", "KC": "#311D8B",
    "CU": "#3025CE", "CS": "#274BFC", "EP": "#648FFF", "KN": "#867A08",
    "PO": "#472C30", "UN": "#9C8975",
}

PITCH_NAMES = {
    "FF": "4-Seam Fastball", "FA": "Fastball",       "SI": "Sinker",
    "FC": "Cutter",          "CH": "Changeup",        "FS": "Splitter",
    "SC": "Screwball",       "FO": "Forkball",        "SL": "Slider",
    "ST": "Sweeper",         "SV": "Slurve",          "KC": "Knuckle Curve",
    "CU": "Curveball",       "CS": "Slow Curve",      "EP": "Eephus",
    "KN": "Knuckleball",     "PO": "Pitch Out",       "UN": "Unknown",
}
