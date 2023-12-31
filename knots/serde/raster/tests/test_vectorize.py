import math

import pytest

from knots.serde.raster.vectorize import Vec2D


class TestVec2D:
    @pytest.mark.parametrize(
        ("vector", "expected_length"),
        [(Vec2D(0, 1), 1), (Vec2D(-1, 0), 1), (Vec2D(1, 1), 2**0.5)],
    )
    def test_length(self, vector: Vec2D, expected_length: float):
        observed_length = vector.length()
        assert observed_length == expected_length

    @pytest.mark.parametrize(
        ("vec1", "vec2", "expected_angle"),
        [
            (Vec2D(0, 1), Vec2D(1, 0), math.pi / 2),
            (Vec2D(0, 1), Vec2D(2, 0), math.pi / 2),
            (Vec2D(1, 0), Vec2D(0, 1), math.pi / 2),
            (Vec2D(1, 0), Vec2D(0, -1), math.pi / 2),
            (Vec2D(1, 0), Vec2D(1, 0), 0),
        ],
    )
    def test_angle_between_radians(
        self, vec1: Vec2D, vec2: Vec2D, expected_angle: float
    ):
        observed_angle = vec1.angle_between_radians(vec2)
        assert observed_angle == expected_angle

    def test_angle_between_degress(self):
        observed_angle = Vec2D(0, 1).angle_between_degrees(Vec2D(1, 0))
        assert observed_angle == 90
