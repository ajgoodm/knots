import math

import pytest
from shapely import LineString, Point

from knots.serde.raster.vectorize import Vec2D, KnotStrand, KnotStrandCollection


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


class TestKnotStrand:
    @pytest.fixture
    def strand(self) -> KnotStrand:
        strand = LineString([(x, x) for x in range(-10, 10)])
        return KnotStrand(strand)

    def test_strand_properties(self, strand: KnotStrand):
        assert strand.length == 20
        assert strand._end_length == 2  # pylint: disable=protected-access

        obs_left, obs_left_vec = strand.left_end
        assert obs_left.equals(Point(-10, -10))
        assert obs_left_vec == Vec2D(-1, -1)

        obs_right, obs_right_vec = strand.right_end
        assert obs_right.equals(Point(9, 9))
        assert obs_right_vec == Vec2D(1, 1)


class TestKnotStrandCollection:
    @pytest.fixture
    def figure_eight(self) -> KnotStrand:
        """A single un-knot crossed on top of itself
        like a figure eight.

            +----+
            |    |
            +--->|<---+
                 |    |
                 +----+

        """
        # fmt: off
        strand = LineString([
            (0.1, 0), (0.2, 0), (0.3, 0),
            (0.3, -0.1), (0.3, -0.2), (-0.3, -0.3),
            (-0.2, -0.3), (-0.1, -0.3), (0, -0.3),
            (0, -0.2), (0, -0.1), (0, 0), (0, 0.1), (0, 0.2), (0, 0.3),
            (-0.1, 0.3), (-0.2, 0.3), (-0.3, 0.3),
            (-0.3, 0.2), (-0.3, 0.1), (-0.3, 0),
            (-0.2, 0), (-0.1, 0)
        ])
        # fmt: on
        return KnotStrand(strand)

    def test_new(self, figure_eight: KnotStrand):
        result = KnotStrandCollection.new([figure_eight])
        print(result)
