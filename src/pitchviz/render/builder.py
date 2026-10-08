from typing import Callable
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from manim import *

from pitchviz.config import PITCH_COLORS, PITCH_NAMES
from pitchviz.data.fetch import (
    pitch_data,
    daily_pitches,
    get_player_heights,
    get_pitcher_game_stats,
    get_player_names,
    get_player_headshot,
)
from pitchviz.data.filters import (
    pitches_filter,
    pitches_filter_vs_left,
    pitches_filter_vs_right,
    EVENT_MAP,
    PITCH_DESCRIPTION_MAP,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
)

IN_ZONE = set(range(1, 10))  # Statcast zones 1-9 are the in-zone 3x3 grid

# Depth (ft from home plate's back tip) of the strike zone's own plane —
# pitch endpoints are solved for this same y so a pitch's dot always lands
# exactly where it crosses the zone, not where it crosses the plate's tip.
ZONE_PLATE_Y = 8.5 / 12

# Real MLB baseball radius (~2.9in diameter), so endpoint dots render at
# true scale rather than an arbitrary fixed size.
BALL_RADIUS_FT = 2.9 / 12 / 2

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


def abs_strike_zone(height_inches: float) -> tuple[float, float]:
    """Automated Ball-Strike (ABS) zone bottom/top in feet, per Baseball Savant's
    definition: bottom at 27% of the batter's height, top at 53.5%."""
    return height_inches * 0.27 / 12, height_inches * 0.535 / 12


def pitch_annotation_text(meta: dict) -> str:
    """Build the per-pitch annotation shown during a sequential (at-bat /
    tunnel) render: velocity + pitch name, the pitch's result, and — only
    when this pitch actually ended the at-bat — the at-bat's outcome.

    No "is this the last pitch" bookkeeping is needed: events is already
    only non-null on the row that ends the at-bat (see
    pitches_filter_by_at_bat / pitches_filter_by_pitch_numbers).
    """
    lines = []

    pitch_name = meta.get("pitch_name")
    release_speed = meta.get("release_speed")
    if pitch_name and release_speed is not None:
        lines.append(f"{pitch_name} · {release_speed:.1f} mph")
    elif release_speed is not None:
        lines.append(f"{release_speed:.1f} mph")

    description = meta.get("description")
    if description:
        lines.append(PITCH_DESCRIPTION_MAP.get(description, description))

    events = meta.get("events")
    if events and not pd.isna(events):
        lines.append(f"Outcome: {EVENT_MAP.get(events, events)}")

    return "\n".join(lines)


def box_score_line(box_score: dict) -> str:
    """One-line box score summary: innings pitched/strikeouts/walks/hits/
    runs/earned runs, formatted as "value LABEL" (e.g. "4 K  3 BB").
    innings_pitched is already "N.1"/"N.2" style for partial innings (MLB's
    own gameLog format), passed through as-is."""
    return (
        f"{box_score['innings_pitched']} IP  {box_score['strikeouts']} K  {box_score['walks']} BB  "
        f"{box_score['hits']} H  {box_score['runs']} R  {box_score['earned_runs']} ER"
    )


def pitch_type_breakdown(df: pd.DataFrame) -> list[dict]:
    """Per-pitch-type arsenal stats for the full-outing summary table:
    count, average velocity/spin/break, and the standard rate stats
    Zone% / Chase% / Whiff% — sorted most-thrown first.

    Zone% = in-zone pitches / pitches with a known zone.
    Chase% = swings on out-of-zone pitches / out-of-zone pitches.
    Whiff% = swings-and-misses / swings (not all pitches — distinct from SwStr%).
    Any rate with a zero denominator (e.g. no out-of-zone pitches of a type)
    comes back None rather than dividing by zero.
    """
    rows = []
    for pitch_type, group in df.groupby("pitch_type"):
        zone = group["zone"]
        description = group["description"]

        known_zone = zone.notna()
        in_zone = known_zone & zone.isin(IN_ZONE)
        out_zone = known_zone & ~zone.isin(IN_ZONE)
        swings = description.isin(SWING_DESCRIPTIONS)
        whiffs = description.isin(WHIFF_DESCRIPTIONS)

        out_zone_count = int(out_zone.sum())
        swing_count = int(swings.sum())

        rows.append({
            "pitch_type": pitch_type,
            "name":       PITCH_NAMES.get(pitch_type, pitch_type),
            "count":      len(group),
            "avg_velo":   group["release_speed"].mean() if group["release_speed"].notna().any() else None,
            "avg_spin":   group["release_spin_rate"].mean() if group["release_spin_rate"].notna().any() else None,
            "avg_ivb":    group["pfx_z"].mean() * 12 if group["pfx_z"].notna().any() else None,
            "avg_hvb":    group["pfx_x"].mean() * 12 if group["pfx_x"].notna().any() else None,
            "zone_pct":   (in_zone.sum() / known_zone.sum() * 100) if known_zone.any() else None,
            "chase_pct":  (swings[out_zone].sum() / out_zone_count * 100) if out_zone_count else None,
            "whiff_pct":  (whiffs.sum() / swing_count * 100) if swing_count else None,
        })

    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


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
        self._pitch_meta: list = []
        self._axes: ThreeDAxes | None = None
        self._filter_label: str | None = None
        self._filter_label_color = WHITE
        # Default/fallback strike zone (ft) — today's generic MLB-average
        # zone, used whenever a render spans more than one batter (full
        # outing, vs-left/vs-right splits) and there's no single batter to
        # size an ABS zone to.
        self._sz_bottom: float = 12 / 12
        self._sz_top: float = (12 + 20) / 12
        # Full-outing-only summary (box score + pitch-arsenal breakdown),
        # shown once the trajectory tails fade in a non-sequential render.
        # {"box_score": dict | None, "pitch_type_rows": list[dict]} or None.
        self._game_summary_data: dict | None = None

    # ------------------------------------------------------------------
    # Builder steps
    # ------------------------------------------------------------------

    def _resolve_strike_zone(self, df: pd.DataFrame) -> None:
        """Size the strike zone to the batter's ABS zone when the filtered
        df is scoped to a single batter (at-bat / tunnel views); otherwise
        leaves the generic default in place."""
        if "batter" not in df.columns:
            return
        batters = df["batter"].dropna().unique()
        if len(batters) != 1:
            return

        batter_id = int(batters[0])
        try:
            heights = get_player_heights([batter_id])
        except Exception:
            return

        height_inches = heights.get(batter_id)
        if height_inches is not None:
            self._sz_bottom, self._sz_top = abs_strike_zone(height_inches)

    def _resolve_game_summary(self, raw_df: pd.DataFrame, filtered_df: pd.DataFrame, filter) -> None:
        """Builds the full-outing summary (box score + pitch-arsenal
        breakdown) shown once the trajectory tails fade — only meaningful
        for the whole-outing scope (pitches_filter), not at-bat/tunnel/
        handedness-split renders, which don't represent a complete game.
        """
        if filter is not pitches_filter or filtered_df.empty:
            return

        player_id = None
        date = None
        try:
            player_id = int(raw_df["pitcher"].iloc[0])
            date = str(raw_df["game_date"].iloc[0])
        except Exception:
            pass

        box_score = None
        try:
            game_pk = int(raw_df["game_pk"].iloc[0])
            season = int(date[:4])
            box_score = get_pitcher_game_stats(player_id, season, game_pk)
        except Exception:
            pass

        pitcher_name = None
        try:
            pitcher_name = get_player_names([player_id]).get(player_id)
        except Exception:
            pass

        headshot = None
        try:
            headshot = get_player_headshot(player_id)
        except Exception:
            pass

        try:
            pitch_type_rows = pitch_type_breakdown(filtered_df)
        except Exception:
            pitch_type_rows = []

        if box_score or pitch_type_rows:
            self._game_summary_data = {
                "box_score": box_score,
                "pitch_type_rows": pitch_type_rows,
                "pitcher_name": pitcher_name,
                "date": date,
                "headshot": headshot,
            }

    def load_pitches(self, date: str, pitcher: str, filter: Callable[[pd.DataFrame], pd.DataFrame]) -> "VizualizationBuilder":
        """Fetch Statcast data and build the parametric curves"""

        self._pitches.clear()
        self._end_points.clear()
        self._end_times.clear()
        self._colors.clear()
        self._pitch_meta.clear()

        raw_df = pitch_data(start_dt=date, pitcher=pitcher)

        # Apply filter
        df = filter(raw_df)
        self._resolve_strike_zone(df)
        self._resolve_game_summary(raw_df, df, filter)

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
        elif hasattr(filter, "_label"):
            self._filter_label = filter._label
            self._filter_label_color = WHITE
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
                    lambda t: position(t, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)[1] - ZONE_PLATE_Y,
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
            self._pitch_meta.append({
                "release_speed": row.get("release_speed"),
                "pitch_name":    row.get("pitch_name"),
                "description":   row.get("description"),
                "events":        row.get("events"),
            })

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
        self._pitch_meta.clear()

        raw_df = df
        df = filter(df)
        self._resolve_strike_zone(df)
        self._resolve_game_summary(raw_df, df, filter)

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
        elif hasattr(filter, "_label"):
            self._filter_label = filter._label
            self._filter_label_color = WHITE
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
                    lambda t: position(t, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)[1] - ZONE_PLATE_Y,
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
            self._pitch_meta.append({
                "release_speed": row.get("release_speed"),
                "pitch_name":    row.get("pitch_name"),
                "description":   row.get("description"),
                "events":        row.get("events"),
            })

        if not self._pitches:
            self._axes = None

        return self

    def buildm_pitches(
        self,
        sequential: bool = False,
        camera: str = "catcher",
        clear_between_pitches: bool = False,
    ) -> type[ThreeDScene]:
        """Return a Manim ThreeDScene class of pitches ready to be rendered.

        sequential: when True, play each pitch's Create animation one at a
        time — with a text annotation (velocity, pitch result, and the
        at-bat's outcome on the pitch that ends it) — then replay all of
        them together overlaid as a closing summary. When False (default),
        all pitches animate simultaneously with no annotations.
        camera: "catcher" (default, looking from behind home plate toward the
        mound) or "mound" (looking from behind the pitcher's mound toward
        home plate).
        clear_between_pitches: only meaningful when sequential=True. When
        True, each pitch's trajectory/dot/annotation is faded out before the
        next one is shown (a true one-at-a-time walkthrough). When False
        (default), they accumulate during the walkthrough, matching the
        pre-existing sequential behavior. Either way the closing overlaid
        replay is unaffected.
        """

        if self._axes is None:
            raise RuntimeError("Call load_pitches() before build().")

        axes         = self._axes
        scale        = self.SCALE
        pitches      = list(self._pitches)
        end_points   = list(self._end_points)
        end_times    = list(self._end_times)
        colors       = list(self._colors)
        pitch_meta   = list(self._pitch_meta)
        filter_label = self._filter_label
        filter_label_color = self._filter_label_color
        sz_bottom_ft = self._sz_bottom
        sz_top_ft    = self._sz_top
        game_summary_data = self._game_summary_data

        class PitchTrajectory(ThreeDScene):
            def construct(self):

                # Background grid — checkered squares lying flat on the
                # ground, bounded to fair territory (between the foul
                # lines, i.e. |x| <= y) rather than a full rectangle that
                # spills into foul territory. Built from individual line
                # segments rather than NumberPlane, which can't be clipped
                # to a wedge shape. Note: a perfectly level camera
                # (catcher's view, phi=90) views this near edge-on,
                # compressing it toward the horizon — a real perspective
                # effect, not a sizing issue.
                grid_step = 7    # ft, larger = fewer/bigger squares
                grid_depth = 70  # ft from home plate
                grid_lines = []
                for y in range(grid_step, grid_depth + 1, grid_step):
                    grid_lines.append(Line3D(
                        start=axes.c2p(-y, y, 0),
                        end=axes.c2p(y, y, 0),
                        thickness=0.01,
                        color=BLUE,
                    ))
                for x in range(-grid_depth, grid_depth + 1, grid_step):
                    y_start = abs(x)
                    if y_start >= grid_depth:
                        continue
                    grid_lines.append(Line3D(
                        start=axes.c2p(x, y_start, 0),
                        end=axes.c2p(x, grid_depth, 0),
                        thickness=0.01,
                        color=BLUE,
                    ))
                grid = VGroup(*grid_lines)
                grid.set_opacity(0.4)

                # Strike Zone Metrics — ABS zone (27%/53.5% of batter height)
                # when the render is scoped to a single batter, otherwise the
                # generic default set in VizualizationBuilder.__init__.
                sz_width  = 17 / 12          # in -> ft
                sz_bottom = sz_bottom_ft
                sz_top    = sz_top_ft
                sz_mid_z  = (sz_bottom + sz_top) / 2

                strike_zone = Rectangle(
                    width=sz_width * scale,
                    height=(sz_top - sz_bottom) * scale,
                )
                strike_zone.move_to(axes.c2p(0, ZONE_PLATE_Y, sz_mid_z))
                strike_zone.rotate(90 * DEGREES, axis=RIGHT)
                strike_zone.set_stroke(WHITE, 4)
                strike_zone.set_fill(opacity=0)

                # Camera
                if camera == "mound":
                    # focal_distance is set explicitly here (default is 20)
                    # because this view's frame_center sits close to the mound
                    # itself in depth; with the default focal_distance, the
                    # perspective projection's factor = focal_distance /
                    # (focal_distance - z) blows up near that asymptote,
                    # ballooning the mound and squishing everything else. A
                    # large focal_distance also flattens perspective falloff
                    # with depth, which is what gives this view its
                    # telephoto-compressed "broadcast center-field camera"
                    # look (full mound + release point in frame, not a
                    # tight first-person close-up). phi is kept close to 90
                    # (near-level) so the shot reads as an elevated-platform
                    # height rather than a steep aerial look-down at the
                    # mound; frame_center z is the elevation of that
                    # platform — counterintuitively, a *larger* z here pushes
                    # the (ground-level) mound further toward the bottom of
                    # frame rather than higher, since raising this near-level
                    # camera's eye height raises the horizon line with it.
                    self.set_camera_orientation(
                        phi=88 * DEGREES,
                        theta=90 * DEGREES,
                        zoom=0.95,
                        focal_distance=150,
                        frame_center=axes.c2p(0, 24, 8),
                    )
                else:
                    # Camera (catcher's POV) — same as the last committed
                    # version.
                    self.set_camera_orientation(
                        phi=90 * DEGREES,
                        theta=-90 * DEGREES,
                        zoom=0.2,
                        frame_center=axes.c2p(0, 30, 3),
                    )

                # Pitcher's mound (18ft-diameter dirt circle centered 59ft from
                # home plate, rubber at the official 60.5ft distance). Flat on
                # the ground, like home_plate/foul lines above — no rotation.
                mound = Circle(radius=9 * scale)
                mound.move_to(axes.c2p(0, 59, 0))
                mound.set_stroke(width=0)
                mound.set_fill(color="#8B5A2B", opacity=0.9)

                rubber = Rectangle(width=2 * scale, height=0.5 * scale)
                rubber.move_to(axes.c2p(0, 60.5, 0.02))
                rubber.set_stroke(width=0)
                rubber.set_fill(color=WHITE, opacity=1)

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

                scene_objects = [grid, strike_zone, right_foul_line, left_foul_line, home_plate, mound, rubber]
                if filter_label is not None:
                    label = Text(filter_label, color=filter_label_color)
                    label.scale(0.3)
                    label.move_to(axes.c2p(0, 0, sz_top + 2.5))
                    label.rotate(90 * DEGREES, axis=RIGHT)
                    if camera == "mound":
                        # The label is a flat 2D VMobject with no real
                        # backface culling, so unlike the symmetric shapes
                        # around it (strike zone, mound, foul lines), its
                        # left-right asymmetry exposes the fact that the
                        # mound camera's theta (90°) is mirrored relative to
                        # the catcher camera's theta (-90°) — without this,
                        # the text reads backwards from the mound's side.
                        label.stretch(-1, 0)
                    scene_objects.append(label)
                self.add(*scene_objects)

                if sequential:
                    # Play each pitch one at a time, with a text annotation
                    # (velocity, pitch result, and — on the pitch that ends
                    # the at-bat — its outcome), then clear and replay all
                    # overlaid as a closing summary.
                    def _make_annotation(meta):
                        text = pitch_annotation_text(meta)
                        if not text:
                            return None
                        ann = Text(text, color=WHITE)
                        ann.scale(0.25)
                        # Between the zone top and the filter label further
                        # above (sz_top + 2.5) — low enough overlaps home
                        # plate's ground outline instead.
                        ann.move_to(axes.c2p(0, 0, sz_top + 1.0))
                        ann.rotate(90 * DEGREES, axis=RIGHT)
                        if camera == "mound":
                            # Same left-right mirror fix as the filter label
                            # above — flat text, no backface culling.
                            ann.stretch(-1, 0)
                        return ann

                    accumulated_pitches = []
                    accumulated_dots = []
                    annotation = None
                    for pitch, t_end, end_point, color, meta in zip(
                        pitches, end_times, end_points, colors, pitch_meta,
                    ):
                        self.play(Create(pitch, run_time=t_end))
                        dot = Dot3D(point=end_point, radius=BALL_RADIUS_FT * scale, color=color)
                        self.add(dot)
                        accumulated_pitches.append(pitch)
                        accumulated_dots.append(dot)

                        new_annotation = _make_annotation(meta)
                        if annotation is not None:
                            self.remove(annotation)
                        if new_annotation is not None:
                            self.add(new_annotation)
                        annotation = new_annotation

                        self.wait(1)

                        if clear_between_pitches:
                            fade_targets = [pitch, dot]
                            if annotation is not None:
                                fade_targets.append(annotation)
                            self.play(*[FadeOut(m) for m in fade_targets])
                            accumulated_pitches.clear()
                            accumulated_dots.clear()
                            annotation = None

                    self.wait(0.5)
                    remaining = [*accumulated_pitches, *accumulated_dots]
                    if annotation is not None:
                        remaining.append(annotation)
                    if remaining:
                        self.play(*[FadeOut(m) for m in remaining])

                    replay_pitches = [pitch.copy() for pitch in pitches]
                    animations = [
                        Create(pitch, run_time=t_end)
                        for pitch, t_end in zip(replay_pitches, end_times)
                    ]
                    self.play(*animations)
                    for end_point, color in zip(end_points, colors):
                        self.add(Dot3D(point=end_point, radius=BALL_RADIUS_FT * scale, color=color))
                    self.wait()
                else:
                    # Animate all pitches simultaneously, then — full-outing
                    # scope only — fade the trajectory tails and show the
                    # box-score / pitch-arsenal summary while the
                    # pitch-location dots stay on screen.
                    animations = [
                        Create(pitch, run_time=t_end)
                        for pitch, t_end in zip(pitches, end_times)
                    ]
                    self.play(*animations)
                    for end_point, color in zip(end_points, colors):
                        self.add(Dot3D(point=end_point, radius=BALL_RADIUS_FT * scale, color=color))

                    if game_summary_data:
                        self.wait(0.5)
                        self.play(*[FadeOut(pitch) for pitch in pitches])

                        box_score = game_summary_data.get("box_score")
                        pitch_rows = game_summary_data.get("pitch_type_rows") or []
                        pitcher_name = game_summary_data.get("pitcher_name")
                        date = game_summary_data.get("date")
                        headshot = game_summary_data.get("headshot")

                        def _pct(v):
                            return f"{v:.0f}%" if v is not None else "-"

                        def _num(v, fmt):
                            return format(v, fmt) if v is not None else "-"

                        # Build everything flat (normal 2D layout, top
                        # element first) and rotate it into a billboard as
                        # one rigid group at the end — the same "stand it
                        # up" trick the filter label and pitch annotations
                        # use.
                        top_level = []  # header text + box-score text + the table
                        reveal_steps = []  # finer-grained top-to-bottom reveal order

                        if pitcher_name or date:
                            header_bits = [b for b in (pitcher_name, date) if b]
                            header_text = Text("   ".join(header_bits), color=WHITE, weight=BOLD).scale(0.3)
                            top_level.append(header_text)
                            reveal_steps.append(header_text)

                        if box_score:
                            box_score_text = Text(box_score_line(box_score), color=WHITE).scale(0.28)
                            top_level.append(box_score_text)
                            reveal_steps.append(box_score_text)

                        table = None
                        if pitch_rows:
                            headers = ["Pitch", "Count", "Velo", "Spin", "IVB", "HVB", "Zone%", "Chase%", "Whiff%"]
                            table_data = [
                                [
                                    r["name"],
                                    str(r["count"]),
                                    _num(r["avg_velo"], ".1f"),
                                    _num(r["avg_spin"], ".0f"),
                                    _num(r["avg_ivb"], ".1f"),
                                    _num(r["avg_hvb"], ".1f"),
                                    _pct(r["zone_pct"]),
                                    _pct(r["chase_pct"]),
                                    _pct(r["whiff_pct"]),
                                ]
                                for r in pitch_rows
                            ]
                            table = Table(
                                table_data,
                                col_labels=[Text(h, weight=BOLD) for h in headers],
                                include_outer_lines=True,
                                element_to_mobject=Text,
                                h_buff=0.3,
                                v_buff=0.25,
                            )
                            table.scale(0.17)
                            # Color each row's pitch-name cell to match its
                            # trajectories/dots (row 1 is the header; column
                            # 1 is "Pitch").
                            for i, r in enumerate(pitch_rows):
                                table.get_entries((i + 2, 1)).set_color(
                                    ManimColor(PITCH_COLORS.get(r["pitch_type"], PITCH_COLORS["UN"]))
                                )

                            # Build the per-cell black fill polygons now,
                            # while the table is still flat/unrotated.
                            # table.get_cell() derives each cell's rectangle
                            # from the table's internal row/column metrics
                            # and does NOT account for any rotation already
                            # applied to the table — building these after
                            # summary.rotate() runs (further below) produces
                            # degenerate, near-zero-height rectangles that
                            # don't actually cover their row, which is why
                            # pitch dots could still show through. Attaching
                            # them as children of `table` here means they
                            # inherit the same move_to/rotate/stretch as
                            # everything else in `summary`.
                            n_rows, n_cols = len(pitch_rows) + 1, len(headers)
                            cell_fills = VGroup()
                            for r in range(1, n_rows + 1):
                                for c in range(1, n_cols + 1):
                                    cell = table.get_cell((r, c))
                                    cell.set_fill(BLACK, opacity=1)
                                    cell.set_stroke(width=0)
                                    cell_fills.add(cell)
                            table.add(cell_fills)

                            top_level.append(table)

                        info_block = VGroup(*top_level).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

                        headshot_img = None
                        if headshot is not None:
                            headshot_img = ImageMobject(headshot)
                            headshot_img.height = info_block.height
                            summary = Group(headshot_img, info_block).arrange(RIGHT, aligned_edge=UP, buff=0.5)
                        else:
                            summary = info_block

                        summary.move_to(axes.c2p(0, 0, sz_top + 1.7))
                        summary.rotate(90 * DEGREES, axis=RIGHT)
                        if camera == "mound":
                            # Same left-right mirror fix as the filter label
                            # and pitch annotations — flat text/image, no
                            # backface culling, so the two mirrors (this one
                            # and the mound camera's own) cancel out.
                            summary.stretch(-1, 0)

                        # Pitch dots/trajectories default to z_index=0;
                        # ThreeDCamera draws non-shaded mobjects (which is
                        # everything here — Dot3D included, it doesn't set
                        # shade_in_3d) in z_index order regardless of actual
                        # 3D depth or add() order, so this alone guarantees
                        # the whole summary — including the table — always
                        # renders in front of the pitches, never obstructed.
                        summary.set_z_index(10)

                        if headshot_img is not None:
                            reveal_steps.insert(0, headshot_img)

                        for step in reveal_steps:
                            self.play(FadeIn(step, rate_func=smooth), run_time=0.9)

                        if table is not None:
                            # Draw the empty table (grid lines + the black
                            # cell fills built earlier, above) right before
                            # filling in the stats; each row's text (header
                            # first) then fades in on top of it, one row at
                            # a time. FadeIn with lag_ratio=0 (rather than
                            # Create) so all ~90 cell/line pieces appear
                            # together instead of drawing in one-by-one.
                            table_shell = VGroup(cell_fills, *table.get_horizontal_lines(), *table.get_vertical_lines())
                            self.play(FadeIn(table_shell, lag_ratio=0, rate_func=smooth), run_time=0.35)

                            for row in table.get_rows():
                                self.play(FadeIn(row, rate_func=smooth), run_time=0.9)

                        self.wait(3)
                    else:
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
