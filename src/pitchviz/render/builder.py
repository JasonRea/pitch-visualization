import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm as sp_norm
from manim import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage
import requests
from io import BytesIO
import textwrap

from pitchviz.config import PITCH_COLORS
from pitchviz.data.fetch import pitch_data, daily_pitches
from pitchviz.data.filters import (
    high_heat_filter,
    absolute_missiles_filter,
    big_five_filter,
    pitches_filter_vs_left,
    pitches_filter_vs_right,
)
from pitchviz.render.assets import player_headshot, player_bio


@dataclass
class PillStyle:
    """
    Describes how a single stat-table column pill should be rendered.

    bg_source: "lookup"   — color comes from lookup_map keyed by raw[j]
               "gradient" — color is a percentile heat map vs. the day's distribution
               "fixed"    — color is always fixed_bg
    """
    bg_source:   str
    lookup_map:  dict | None = None
    fixed_bg:    str         = "#f5f5f5"
    text_color:  str         = "#1a1a1a"
    font_weight: str         = "normal"
    font_size:   float       = 10.0


@dataclass
class GraphicConfig:
    """
    All graphic-specific configuration for buildp().

    Callers define the data pipeline (filter_fn, player_id_fn, row_data_fn),
    the visual structure (col_labels, pill_styles, title copy), and the
    optional per-row description (description_fn, None for most graphics).
    """
    # Title block
    title:     str
    subtitle:  str
    emoji_url: str

    # Table structure
    col_labels:  list[str]
    pill_styles: list[PillStyle]

    # Data pipeline
    filter_fn:    Callable                  # (full_df) -> filtered_df
    player_id_fn: Callable                  # (filtered_df, i) -> int player ID
    row_data_fn:  Callable                  # (filtered_df, i) -> (formatted: list[str], raw: list)

    # col_index -> callable (full_df) -> Series, for gradient columns
    stat_series_fns: dict[int, Callable]

    # Optional per-row play description (only big_five uses this)
    description_fn: Callable | None = None

    # Output
    output_prefix: str = "graphic"
    empty_error:   str = "No data found for {date}."


EVENT_COLORS = {
    "Home Run":         "#C41E3A",
    "Triple":           "#7B2D8B",
    "Double":           "#1565C0",
    "Single":           "#2E7D32",
    "Flyout":           "#555555",
    "Strikeout":        "#1a1a1a",
    "GIDP":             "#795548",
    "Force Out":        "#546E7A",
    "Fielder's Choice": "#546E7A",
    "Sac Bunt":         "#78909C",
    "Hit By Pitch":     "#E65100",
    "walk":             "#558B2F",
    "Sac Fly":          "#78909C",
    "Double Play":      "#795548",
    "Error":            "#F57F17",
}

# TODO Specify specific pitch types, vs rhb, lhb, vs particular batters (IN DEVELOPMENT)

def position(
    t: float,
    x0: float, y0: float, z0: float,
    vx0: float, vy0: float, vz0: float,
    ax: float, ay: float, az: float,
) -> np.ndarray:
    x = x0 + vx0 * t + 0.5 * ax * t ** 2
    y = y0 + vy0 * t + 0.5 * ay * t ** 2
    z = z0 + vz0 * t + 0.5 * az * t ** 2
    return np.array([x, y, z])

class VizualizationBuilder:
    """
    Builder for Manim pitch trajectory visualizations.

    Usage:
        scene_class = (
            VizualizationBuilder()
            .load_pitches(date="2026-02-24", pitcher="Ranger Suarez")
            .buildm_pitches()
        )
        VizualizationBuilder.render(scene_class, quality="high_quality", filename=f"{date} {pitcher}")
    """

    # Manim units per foot
    SCALE: float = 6 / 10

    def __init__(self):
        self._pitches: list = []
        self._end_points: list = []
        self._end_times: list = []
        self._colors: list = []
        self._axes: ThreeDAxes | None = None
        self._filter_label: str | None = None
        self._filter_label_color = WHITE

    # ------------------------------------------------------------------
    # Builder steps
    # ------------------------------------------------------------------

    def load_pitches(self, date: str, pitcher: str, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> "VizualizationBuilder":
        """Fetch Statcast data and build the parametric curves"""

        self._pitches.clear()
        self._end_points.clear()
        self._end_times.clear()
        self._colors.clear()

        df = pitch_data(start_dt=date, pitcher=pitcher)

        # Apply filter
        df = filter(df)

        if filter is pitches_filter_vs_left:
            self._filter_label = "vs Left"
            self._filter_label_color = WHITE
        elif filter is pitches_filter_vs_right:
            self._filter_label = "vs Right"
            self._filter_label_color = WHITE
        elif hasattr(filter, "_pitch_type"):
            pitch_name = df["pitch_name"].iloc[0] if not df.empty else filter._pitch_type
            self._filter_label = pitch_name
            self._filter_label_color = ManimColor(PITCH_COLORS.get(filter._pitch_type, PITCH_COLORS["UN"]))
        else:
            self._filter_label = None
            self._filter_label_color = WHITE

        self._axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[0, 60.5, 10],
            z_range=[0, 20, 1],
            x_length=self.SCALE * 10,
            y_length=self.SCALE * 60.5,
            z_length=self.SCALE * 20,
        )

        for _, row in df.iterrows():
            x0  = float(row["release_pos_x"])
            y0  = float(row["release_pos_y"])
            z0  = float(row["release_pos_z"])
            vx0 = float(row["vx0"])
            vy0 = float(row["vy0"])
            vz0 = float(row["vz0"])
            ax  = float(row["ax"])
            ay  = float(row["ay"])
            az  = float(row["az"])
            pitch_type = str(row["pitch_type"])

            try:
                t_end = brentq(
                    lambda t: position(t, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)[1],
                    0, 1.5,
                )
            except ValueError:
                print(f"Skipping pitch: brentq failed (y0={y0:.2f}, vy0={vy0:.2f}, ay={ay:.2f})")
                continue

            pitch = ParametricFunction(
                lambda t,
                    x0=x0, y0=y0, z0=z0,
                    vx0=vx0, vy0=vy0, vz0=vz0,
                    ax=ax, ay=ay, az=az,
                    t_end=t_end:
                    self._axes.c2p(*position(t * t_end, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)),
                t_range=[0, 1],
                stroke_width=2,
                color=PITCH_COLORS.get(pitch_type, PITCH_COLORS["UN"]),
            )

            self._pitches.append(pitch)
            self._end_points.append(
                self._axes.c2p(*position(t_end, x0, y0, z0, vx0, vy0, vz0, ax, ay, az))
            )
            self._end_times.append(t_end)
            self._colors.append(PITCH_COLORS.get(pitch_type, PITCH_COLORS["UN"]))

        if not self._pitches:
            self._axes = None

        return self

    def load_pitches_from_df(
        self,
        df: pd.DataFrame,
        filter: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> "VizualizationBuilder":
        """Build parametric curves from an already-fetched Statcast DataFrame."""

        self._pitches.clear()
        self._end_points.clear()
        self._end_times.clear()
        self._colors.clear()

        df = filter(df)

        if filter is pitches_filter_vs_left:
            self._filter_label = "vs Left"
            self._filter_label_color = WHITE
        elif filter is pitches_filter_vs_right:
            self._filter_label = "vs Right"
            self._filter_label_color = WHITE
        elif hasattr(filter, "_pitch_type"):
            pitch_name = df["pitch_name"].iloc[0] if not df.empty else filter._pitch_type
            self._filter_label = pitch_name
            self._filter_label_color = ManimColor(PITCH_COLORS.get(filter._pitch_type, PITCH_COLORS["UN"]))
        else:
            self._filter_label = None
            self._filter_label_color = WHITE

        if df.empty:
            self._axes = None
            return self

        self._axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[0, 60.5, 10],
            z_range=[0, 20, 1],
            x_length=self.SCALE * 10,
            y_length=self.SCALE * 60.5,
            z_length=self.SCALE * 20,
        )

        for _, row in df.iterrows():
            x0  = float(row["release_pos_x"])
            y0  = float(row["release_pos_y"])
            z0  = float(row["release_pos_z"])
            vx0 = float(row["vx0"])
            vy0 = float(row["vy0"])
            vz0 = float(row["vz0"])
            ax  = float(row["ax"])
            ay  = float(row["ay"])
            az  = float(row["az"])
            pitch_type = str(row["pitch_type"])

            try:
                t_end = brentq(
                    lambda t: position(t, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)[1],
                    0, 1.5,
                )
            except ValueError:
                print(f"Skipping pitch: brentq failed (y0={y0:.2f}, vy0={vy0:.2f}, ay={ay:.2f})")
                continue

            pitch = ParametricFunction(
                lambda t,
                    x0=x0, y0=y0, z0=z0,
                    vx0=vx0, vy0=vy0, vz0=vz0,
                    ax=ax, ay=ay, az=az,
                    t_end=t_end:
                    self._axes.c2p(*position(t * t_end, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)),
                t_range=[0, 1],
                stroke_width=2,
                color=PITCH_COLORS.get(pitch_type, PITCH_COLORS["UN"]),
            )

            self._pitches.append(pitch)
            self._end_points.append(
                self._axes.c2p(*position(t_end, x0, y0, z0, vx0, vy0, vz0, ax, ay, az))
            )
            self._end_times.append(t_end)
            self._colors.append(PITCH_COLORS.get(pitch_type, PITCH_COLORS["UN"]))

        if not self._pitches:
            self._axes = None

        return self

    def buildm_pitches(self) -> type[ThreeDScene]:
        """Return a Manim ThreeDScene class of pitches ready to be rendered."""

        if self._axes is None:
            raise RuntimeError("Call load_pitches() before build().")

        axes         = self._axes
        scale        = self.SCALE
        pitches      = list(self._pitches)
        end_points   = list(self._end_points)
        end_times    = list(self._end_times)
        colors       = list(self._colors)
        filter_label = self._filter_label
        filter_label_color = self._filter_label_color

        class PitchTrajectory(ThreeDScene):
            def construct(self):

                # Background grid
                grid = NumberPlane(
                    x_range=[-5, 5, 1],
                    y_range=[0, 20, 1],
                    x_length=scale * 10,
                    y_length=scale * 20,
                    background_line_style={
                        "stroke_color": BLUE,
                        "stroke_width": 1,
                        "stroke_opacity": 0.4,
                    },
                    axis_config={"stroke_opacity": 0},
                )
                grid.move_to(axes.c2p(0, 0, 10))
                grid.rotate(90 * DEGREES, axis=RIGHT)

                # Strike Zone Metrics
                sz_width  = 17 / 12          # in -> ft
                sz_bottom = 12 / 12
                sz_top    = (12 + 20) / 12
                sz_mid_z  = (sz_bottom + sz_top) / 2

                strike_zone = Rectangle(
                    width=sz_width * scale,
                    height=(sz_top - sz_bottom) * scale,
                )
                strike_zone.move_to(axes.c2p(0, (8.5/12), sz_mid_z))
                strike_zone.rotate(90 * DEGREES, axis=RIGHT)
                strike_zone.set_stroke(WHITE, 4)
                strike_zone.set_fill(opacity=0)

                # Camera (catcher's POV)
                self.set_camera_orientation(
                    phi=90 * DEGREES,
                    theta=-90 * DEGREES,
                    zoom=0.2,
                    frame_center=axes.c2p(0, 30, 3),
                )

                # Foul lines (XY plane, z=0): y=x right, y=-x left
                foul_extent = 330  # feet
                right_foul_line = Line3D(
                    start=axes.c2p(1, 1, 0),
                    end=axes.c2p(foul_extent, foul_extent, 0),
                    thickness=0.02,
                    color=WHITE,
                )
                left_foul_line = Line3D(
                    start=axes.c2p(-1, 1, 0),
                    end=axes.c2p(-foul_extent, foul_extent, 0),
                    thickness=0.02,
                    color=WHITE,
                )

                # Home plate (irregular pentagon, tip at origin, flat edge toward pitcher)
                home_plate = Polygon(
                    axes.c2p(0,         0,        0),  # tip
                    axes.c2p( 8.5/12,  8.5/12,   0),  # right back
                    axes.c2p( 8.5/12,  17/12,    0),  # right front
                    axes.c2p(-8.5/12,  17/12,    0),  # left front
                    axes.c2p(-8.5/12,  8.5/12,   0),  # left back
                )
                home_plate.set_stroke(WHITE, 2)
                home_plate.set_fill(opacity=0)

                scene_objects = [grid, strike_zone, right_foul_line, left_foul_line, home_plate]
                if filter_label is not None:
                    label = Text(filter_label, color=filter_label_color)
                    label.scale(0.3)
                    label.move_to(axes.c2p(0, 0, sz_top + 2.5))
                    label.rotate(90 * DEGREES, axis=RIGHT)
                    scene_objects.append(label)
                self.add(*scene_objects)

                # Animate all pitches simultaneously
                animations = [
                    Create(pitch, run_time=t_end)
                    for pitch, t_end in zip(pitches, end_times)
                ]
                self.play(*animations)
                for end_point, color in zip(end_points, colors):
                    self.add(Dot3D(point=end_point, radius=0.05, color=color))
                self.wait()

        return PitchTrajectory

    # ------------------------------------------------------------------
    # Graphic configs
    # ------------------------------------------------------------------

    @staticmethod
    def _high_heat_config() -> "GraphicConfig":
        PITCH_EVENT_COLORS = PITCH_COLORS

        return GraphicConfig(
            title="High Heat",
            subtitle="Top 5 Hardest Pitches",
            emoji_url="https://em-content.zobj.net/source/apple/419/fire_1f525.png",
            col_labels=["Pitch", "Velo (mph)", "Spin (rpm)", "HB (in)", "iVB (in)"],
            pill_styles=[
                PillStyle(bg_source="lookup", lookup_map=PITCH_EVENT_COLORS,
                          text_color="white", font_weight="bold", font_size=7.5),
                PillStyle(bg_source="gradient",
                          text_color="#DA2626", font_weight="bold", font_size=10.0),
                PillStyle(bg_source="gradient",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
                PillStyle(bg_source="gradient",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
                PillStyle(bg_source="gradient",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
            ],
            filter_fn=high_heat_filter,
            player_id_fn=lambda df, i: int(df["pitcher"].iloc[i]),
            row_data_fn=lambda df, i: (
                [
                    df["pitch_name"].iloc[i],
                    f"{df['release_speed'].iloc[i]:.1f}",
                    f"{int(df['release_spin_rate'].iloc[i]):,}",
                    f"{df['pfx_x'].iloc[i] * 12:+.1f}",
                    f"{df['pfx_z'].iloc[i] * 12:+.1f}",
                ],
                [
                    df["pitch_type"].iloc[i],
                    df["release_speed"].iloc[i],
                    float(df["release_spin_rate"].iloc[i]),
                    df["pfx_x"].iloc[i] * 12,
                    df["pfx_z"].iloc[i] * 12,
                ],
            ),
            stat_series_fns={
                1: lambda full_df: full_df["release_speed"],
                2: lambda full_df: full_df["release_spin_rate"],
                3: lambda full_df: full_df["pfx_x"] * 12,
                4: lambda full_df: full_df["pfx_z"] * 12,
            },
            description_fn=None,
            output_prefix="high_heat",
            empty_error="No pitch data found for {date}.",
        )

    @staticmethod
    def _absolute_missiles_config() -> "GraphicConfig":
        return GraphicConfig(
            title="Absolute Missiles",
            subtitle="Top 5 Hardest Hit Fly Balls",
            emoji_url="https://em-content.zobj.net/source/apple/419/rocket_1f680.png",
            col_labels=["Event", "Exit Velo (mph)", "Launch Angle", "Distance (ft)", "xBA"],
            pill_styles=[
                PillStyle(bg_source="lookup", lookup_map=EVENT_COLORS,
                          text_color="white", font_weight="bold", font_size=7.5),
                PillStyle(bg_source="gradient",
                          text_color="#DA2626", font_weight="bold", font_size=10.0),
                PillStyle(bg_source="gradient",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
                PillStyle(bg_source="gradient",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
                PillStyle(bg_source="gradient",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
            ],
            filter_fn=absolute_missiles_filter,
            player_id_fn=lambda df, i: int(df["batter"].iloc[i]),
            row_data_fn=lambda df, i: (
                [
                    df["events"].iloc[i],
                    f"{df['launch_speed'].iloc[i]:.1f}",
                    f"{int(df['launch_angle'].iloc[i]) if pd.notna(df['launch_angle'].iloc[i]) else '--'}°",
                    f"{int(df['hit_distance_sc'].iloc[i]) if pd.notna(df['hit_distance_sc'].iloc[i]) else '--'}",
                    f"{df['estimated_ba_using_speedangle'].iloc[i]:.3f}",
                ],
                [
                    df["events"].iloc[i],
                    df["launch_speed"].iloc[i],
                    float(df["launch_angle"].iloc[i]),
                    float(df["hit_distance_sc"].iloc[i]),
                    float(df["estimated_ba_using_speedangle"].iloc[i]),
                ],
            ),
            stat_series_fns={
                1: lambda full_df: full_df["launch_speed"],
                2: lambda full_df: full_df["launch_angle"],
                3: lambda full_df: full_df["hit_distance_sc"],
                4: lambda full_df: full_df["estimated_ba_using_speedangle"],
            },
            description_fn=None,
            output_prefix="absolute_missiles",
            empty_error="No batted ball data found for {date}.",
        )

    @staticmethod
    def _big_five_config() -> "GraphicConfig":
        def _big_five_row_data(df, i):
            event   = df["events"].iloc[i]
            wpa     = df["delta_home_win_exp"].iloc[i]
            run_exp = df["delta_run_exp"].iloc[i]
            inning  = int(df["inning"].iloc[i])
            formatted = [
                event,
                f"{wpa:.3f}",
                f"+{run_exp:.3f}" if run_exp >= 0 else f"{run_exp:.3f}",
                str(inning),
            ]
            raw = [event, wpa, run_exp, inning]
            return formatted, raw

        return GraphicConfig(
            title="Wheeeee!",
            subtitle="Top 5 Plays by Win Probability Added",
            emoji_url="https://em-content.zobj.net/source/apple/419/trophy_1f3c6.png",
            col_labels=["Event", "WPA", "Run Exp", "Inning"],
            pill_styles=[
                PillStyle(bg_source="lookup", lookup_map=EVENT_COLORS,
                          text_color="white", font_weight="bold", font_size=7.5),
                PillStyle(bg_source="gradient",
                          text_color="#DA2626", font_weight="bold", font_size=10.0),
                PillStyle(bg_source="gradient",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
                PillStyle(bg_source="fixed", fixed_bg="#f5f5f5",
                          text_color="#1a1a1a", font_weight="normal", font_size=10.0),
            ],
            filter_fn=big_five_filter,
            player_id_fn=lambda df, i: (
                int(df["pitcher"].iloc[i])
                if df["events"].iloc[i] == "Strikeout"
                else int(df["batter"].iloc[i])
            ),
            row_data_fn=_big_five_row_data,
            stat_series_fns={
                1: lambda full_df: full_df["delta_home_win_exp"],
                2: lambda full_df: full_df["delta_run_exp"],
            },
            description_fn=lambda df, i: str(df["des"].iloc[i]),
            output_prefix="big_five",
            empty_error="No play data found for {date}.",
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_assets(
        player_ids: list[int],
    ) -> tuple[list, list]:
        """Concurrently fetch headshots and bios for a list of player IDs."""
        n = len(player_ids)
        headshots: list[PILImage.Image | None] = [None] * n
        bios:      list[tuple | None]          = [None] * n

        def _fetch_hs(idx, pid):
            try:    headshots[idx] = player_headshot(str(pid))
            except: pass

        def _fetch_bio(idx, pid):
            try:    bios[idx] = player_bio(str(pid))
            except: pass

        with ThreadPoolExecutor(max_workers=n * 2) as ex:
            futs = (
                [ex.submit(_fetch_hs,  i, pid) for i, pid in enumerate(player_ids)] +
                [ex.submit(_fetch_bio, i, pid) for i, pid in enumerate(player_ids)]
            )
            for f in as_completed(futs):
                f.result()

        return headshots, bios

    @staticmethod
    def _pct_color(value: float, series: pd.Series) -> str:
        """Map *value* to a hex colour via its CDF percentile in *series*."""
        _cmap = mcolors.LinearSegmentedColormap.from_list(
            "stat_heat", ["#648FFF", "#FFFFFF", "#FFB000"]
        )
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty or s.std() == 0:
            return "#ffffff"
        pct = float(sp_norm.cdf(value, loc=s.mean(), scale=s.std()))
        return mcolors.to_hex(_cmap(pct))

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_title(fig, gs_slot, title: str, subtitle_with_date: str,
                    emoji_url: str, dark: str, subtext: str):
        try:
            resp      = requests.get(emoji_url)
            emoji_img = PILImage.open(BytesIO(resp.content)).convert("RGBA")
            emoji_arr = np.array(emoji_img)
        except Exception:
            emoji_arr = None

        ax = fig.add_subplot(gs_slot)
        ax.axis("off")
        ax.text(0.5, 0.70, title,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=22, fontweight="bold", color=dark)
        ax.text(0.5, 0.22, subtitle_with_date,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=subtext)

        if emoji_arr is not None:
            for x_pos in (0.20, 0.76):
                ax_e = ax.inset_axes([x_pos, 0.38, 0.07, 0.58])
                ax_e.imshow(emoji_arr)
                ax_e.axis("off")

    @staticmethod
    def _draw_header(fig, gs_slot, col_labels: list[str],
                     hs_frac: float, dark: str):
        n_cols = len(col_labels)
        col_xs = [hs_frac + (1 - hs_frac) * (j + 0.5) / n_cols for j in range(n_cols)]

        ax = fig.add_subplot(gs_slot)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        for label, cx in zip(col_labels, col_xs):
            ax.text(cx, 0.5, label,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=dark,
                    linespacing=1.2, zorder=2)

        ax.axhline(0, color=dark, linewidth=1, xmin=0, xmax=1)

    @staticmethod
    def _draw_headshot_pane(fig, gs_row_slot, headshot, name: str,
                            row_bg: str, dark: str):
        ax = fig.add_subplot(gs_row_slot)
        ax.set_facecolor(row_bg)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        if headshot is not None:
            inset = ax.inset_axes([0.10, 0.20, 0.80, 0.75])
            inset.imshow(np.array(headshot.convert("RGBA")))
            inset.axis("off")
        else:
            ax.add_patch(plt.Circle((0.50, 0.60), 0.30,
                                    color="#cccccc", transform=ax.transAxes,
                                    zorder=2))

        ax.text(0.50, 0.10, name,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color=dark)
        ax.axhline(0, color=dark, linewidth=0.8)

    @staticmethod
    def _draw_data_pane(fig, gs_row_slot, formatted: list, raw: list,
                        pill_styles: list, stat_series: dict,
                        description: str | None, row_bg: str,
                        dark: str, subtext: str):
        n_cols   = len(formatted)
        col_frac = 1 / n_cols
        pill_w   = col_frac
        pill_h   = 0.44
        pill_y   = 0.28

        ax = fig.add_subplot(gs_row_slot)
        ax.set_facecolor(row_bg)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        for j, (val, style) in enumerate(zip(formatted, pill_styles)):
            cx = (j + 0.5) * col_frac

            if style.bg_source == "lookup":
                bg = style.lookup_map.get(raw[j], "#555555")
            elif style.bg_source == "gradient":
                bg = VizualizationBuilder._pct_color(raw[j], stat_series[j])
            else:  # "fixed"
                bg = style.fixed_bg

            pill = FancyBboxPatch(
                (cx - pill_w / 2, pill_y), pill_w, pill_h,
                boxstyle="square,pad=0",
                facecolor=bg, edgecolor="none",
                transform=ax.transAxes, zorder=1, clip_on=False,
            )
            ax.add_patch(pill)
            ax.text(cx, 0.50, val,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=style.font_size, color=style.text_color,
                    fontweight=style.font_weight, zorder=2)

        if description is not None:
            des_lines   = textwrap.wrap(description, width=100)[:2]
            des_display = "\n".join(des_lines)
            ax.text(0.02, 0.18, des_display,
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=6.5, color=subtext, style="italic",
                    linespacing=1.4, zorder=2)

        ax.axhline(0, color=dark, linewidth=0.8)

    @staticmethod
    def _draw_footer(fig, gs_slot, dark: str, subtext: str):
        ax = fig.add_subplot(gs_slot)
        ax.axis("off")
        ax.axhline(1.0, color=dark, linewidth=1.0)
        ax.text(0.02, 0.45, "By: @baseballfornerds",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=8, color=subtext, style="italic")
        ax.text(0.98, 0.45, "Data: MLB  |  Images: MLB, ESPN",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=8, color=subtext, style="italic")

    # ------------------------------------------------------------------
    # Unified buildp
    # ------------------------------------------------------------------

    def buildp(self, date: str, config: "GraphicConfig"):
        """
        Unified graphic builder. Accepts a GraphicConfig that describes
        every graphic-specific detail (data pipeline, pill styles, copy).

        Example:
            fig = builder.buildp("2025-04-01", VizualizationBuilder._high_heat_config())
        """

        # ── Data ──────────────────────────────────────────────────────
        full_day_df  = daily_pitches(date)
        filtered_df  = config.filter_fn(full_day_df)

        if filtered_df.empty:
            raise RuntimeError(config.empty_error.format(date=date))

        n           = len(filtered_df)
        player_ids  = [config.player_id_fn(filtered_df, i) for i in range(n)]
        headshots, bios = self._fetch_assets(player_ids)

        # Pre-build per-column stat series for gradient pills
        stat_series = {
            j: fn(full_day_df)
            for j, fn in config.stat_series_fns.items()
        }

        # ── Layout constants ──────────────────────────────────────────
        BORDER   = "#e0e0e0"
        DARK     = "#1a1a1a"
        SUBTEXT  = "#666666"
        HS_FRAC  = 0.20
        ROW_H    = 1.3
        TITLE_H  = 0.8
        HDR_H    = 0.40
        FOOTER_H = 0.30

        fig = plt.figure(figsize=(7.2, 9), facecolor="white", dpi=150)
        gs  = gridspec.GridSpec(
            2 + n + 1, 1,
            figure=fig,
            height_ratios=[TITLE_H, HDR_H] + [ROW_H] * n + [FOOTER_H],
            hspace=0,
            left=0.03, right=0.97,
            top=1.0, bottom=0.0,
        )

        # ── Title ─────────────────────────────────────────────────────
        self._draw_title(
            fig, gs[0],
            title=config.title,
            subtitle_with_date=f"{config.subtitle}  ·  {date}",
            emoji_url=config.emoji_url,
            dark=DARK, subtext=SUBTEXT,
        )

        # ── Column headers ────────────────────────────────────────────
        self._draw_header(fig, gs[1], config.col_labels, HS_FRAC, DARK)

        # ── Player rows ───────────────────────────────────────────────
        for i in range(n):
            gs_row = gridspec.GridSpecFromSubplotSpec(
                1, 2,
                subplot_spec=gs[2 + i],
                width_ratios=[HS_FRAC, 1 - HS_FRAC],
                wspace=0,
            )

            self._draw_headshot_pane(
                fig, gs_row[0],
                headshot=headshots[i],
                name=bios[i][0] if bios[i] else "Unknown",
                row_bg=BORDER, dark=DARK,
            )

            formatted, raw = config.row_data_fn(filtered_df, i)
            description    = config.description_fn(filtered_df, i) if config.description_fn else None

            self._draw_data_pane(
                fig, gs_row[1],
                formatted=formatted,
                raw=raw,
                pill_styles=config.pill_styles,
                stat_series=stat_series,
                description=description,
                row_bg=BORDER, dark=DARK, subtext=SUBTEXT,
            )

        # ── Footer ────────────────────────────────────────────────────
        self._draw_footer(fig, gs[2 + n], DARK, SUBTEXT)

        # ── Save ──────────────────────────────────────────────────────
        output_dir = "media/images/plots"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(
            fname=f"{output_dir}/{config.output_prefix}_{date}.png",
            bbox_inches="tight", dpi=150, facecolor="white",
        )

        return fig

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    @staticmethod
    def render(
        scene: type[Scene],
        quality: str | None = None,
        filename: str | None = None,
    ) -> None:
        """
        Render a Manim scene class.

        Args:
            scene:    A Scene subclass (e.g. the return value of build()).
            quality:  One of "low_quality", "medium_quality", "high_quality",
                      "fourk_quality". Defaults to Manim's current config.
            filename: Output filename (without extension).
        """
        if quality:
            config.quality = quality
        if filename:
            config.output_file = filename

        scene().render()
