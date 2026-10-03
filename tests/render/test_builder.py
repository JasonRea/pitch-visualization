import numpy as np
import pytest

from pitchviz.render.builder import position


def test_position_at_t_zero_is_release_point():
    p = position(0, x0=1.0, y0=60.5, z0=5.5, vx0=0, vy0=-130, vz0=-5, ax=0, ay=20, az=-16)

    assert p == pytest.approx(np.array([1.0, 60.5, 5.5]))


def test_position_matches_kinematic_equation_at_t():
    x0, y0, z0 = 0.0, 60.5, 5.5
    vx0, vy0, vz0 = 2.0, -130.0, -4.0
    ax, ay, az = -10.0, 25.0, -18.0
    t = 0.4

    p = position(t, x0, y0, z0, vx0, vy0, vz0, ax, ay, az)

    expected = np.array([
        x0 + vx0 * t + 0.5 * ax * t**2,
        y0 + vy0 * t + 0.5 * ay * t**2,
        z0 + vz0 * t + 0.5 * az * t**2,
    ])
    assert p == pytest.approx(expected)


def test_position_y_decreases_toward_plate_over_time():
    # A pitch travels from the mound (y~60.5) toward home plate (y=0): y should fall.
    p0 = position(0.0,  x0=0, y0=60.5, z0=5.5, vx0=0, vy0=-130, vz0=0, ax=0, ay=20, az=0)
    p1 = position(0.3,  x0=0, y0=60.5, z0=5.5, vx0=0, vy0=-130, vz0=0, ax=0, ay=20, az=0)

    assert p1[1] < p0[1]
