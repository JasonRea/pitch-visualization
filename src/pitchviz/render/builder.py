from typing import Callable
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from manim import *

from pitchviz.config import PITCH_COLORS
from pitchviz.data.fetch import pitch_data, daily_pitches
from pitchviz.data.filters import (
    pitches_filter_vs_left,
    pitches_filter_vs_right,
)

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
