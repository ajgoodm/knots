import math
import operator as ops
from collections import deque
from collections.abc import Iterable
from enum import Enum
from functools import partial
from typing import Optional

import attr
import numpy as np
import numpy.typing as npt
from attr.validators import instance_of, deep_iterable
from shapely.geometry import Point, LineString

from knots.core.knot import Knot

MAX_ANGLE_STRAND_TO_CANDIDATE_DEGREES = 30


@attr.frozen
class Vec2D:
    x_component: float
    y_component: float

    def length(self) -> float:
        return math.sqrt(self.x_component**2 + self.y_component**2)

    def unit_vector(self) -> "Vec2D":
        return Vec2D(self.x_component / self.length(), self.y_component / self.length())

    def dot_product(self, other: "Vec2D") -> float:
        return (
            self.x_component * other.x_component + self.y_component * other.y_component
        )

    def angle_between_radians(self, other: "Vec2D") -> float:
        """Return the angle between self and other in radians.
        This function is symmetric, always returning a positive angle.
        It also always returns the _smallest_ angle between two vectors
        (i.e. is always less than pi)
        """
        self_unit = self.unit_vector()
        other_unit = other.unit_vector()

        unit_dot_product = self_unit.dot_product(other_unit)
        angle_radians = math.acos(unit_dot_product)
        while angle_radians > 2 * math.pi:
            angle_radians -= 2 * math.pi
        while angle_radians < 0:
            angle_radians += 2 * math.pi
        if angle_radians > math.pi:
            # we want the _smallest_ angle between these two vectors
            angle_radians = 2 * math.pi - angle_radians
        return angle_radians

    def angle_between_degrees(self, other: "Vec2D") -> float:
        angle_radians = self.angle_between_radians(other)
        return angle_radians * (360.0 / (2 * math.pi))


@attr.frozen
class KnotStrand:
    MINIMUM_LENGTH = 20

    strand: LineString = attr.ib(validator=instance_of(LineString))

    @strand.validator
    def is_sufficiently_long(self, attribute, value):  # type: ignore[no-untyped-def]
        if len(value.coords) < KnotStrand.MINIMUM_LENGTH:
            raise ValueError(
                f"Knot strands must have at least {KnotStrand.MINIMUM_LENGTH} coordinates"
            )

    @property
    def length(self) -> int:
        return len(self.strand.coords)

    @property
    def _end_length(self) -> int:
        """10 percent of the self's length; used in finding end vectors."""
        return math.ceil(self.length * 0.1)

    @property
    def left_end(self) -> tuple[Point, Vec2D]:
        end = self.strand.coords[: self._end_length]
        from_ = end[-1]
        to_ = end[0]
        vec = Vec2D(
            x_component=to_[0] - from_[0],
            y_component=to_[1] - from_[1],
        )

        return Point(self.strand.coords[0]), vec

    @property
    def right_end(self) -> tuple[Point, Vec2D]:
        end = self.strand.coords[(-1 * self._end_length) :]
        from_ = end[0]
        to_ = end[-1]
        vec = Vec2D(
            x_component=to_[0] - from_[0],
            y_component=to_[1] - from_[1],
        )

        return Point(self.strand.coords[-1]), vec


class _LeftOrRight(Enum):
    LEFT = 0
    RIGHT = 1


_StrandEnd = tuple[KnotStrand, _LeftOrRight]


def _filter_candidate_ends(
    previous_end: _StrandEnd, strand_ends: Iterable[_StrandEnd]
) -> list[_StrandEnd]:
    """Look through remaining unpaired strand ends and filter to
    those that are 'lined up' with the previous end.
    """
    previous_strand, left_or_right = previous_end
    if left_or_right == _LeftOrRight.LEFT:
        previous_point, previous_vec = previous_strand.left_end
    else:
        previous_point, previous_vec = previous_strand.right_end

    candidates: list[_StrandEnd] = []
    for strand_end in strand_ends:
        strand, left_or_right = strand_end

        if left_or_right == _LeftOrRight.LEFT:
            candidate_point, _ = strand.left_end
        else:
            candidate_point, _ = strand.right_end

        vec_to_candidate = Vec2D(
            candidate_point.x - previous_point.x, candidate_point.y - previous_point.y
        )
        angle_to_candidate = previous_vec.angle_between_degrees(vec_to_candidate)
        if angle_to_candidate <= MAX_ANGLE_STRAND_TO_CANDIDATE_DEGREES:
            candidates.append(strand_end)

    return candidates


def _distance_to_candidate(candidate: _StrandEnd, previous_end: _StrandEnd) -> float:
    """Find the distance from the previous end to the candidate strand end"""
    previous_strand, left_or_right = previous_end
    if left_or_right == _LeftOrRight.LEFT:
        from_, _ = previous_strand.left_end
    else:
        from_, _ = previous_strand.right_end

    candidate_strand, left_or_right = candidate
    if left_or_right == _LeftOrRight.LEFT:
        to_, _ = candidate_strand.left_end
    else:
        to_, _ = candidate_strand.right_end

    return Vec2D(x_component=to_.x - from_.x, y_component=to_.y - from_.y).length()


@attr.frozen
class KnotStrandCollection:
    strands: tuple[KnotStrand, ...] = attr.ib(
        validator=deep_iterable(instance_of(KnotStrand), instance_of(tuple))
    )

    @classmethod
    def new(cls, strands: Iterable[KnotStrand]) -> "KnotStrandCollection":
        """Take a collection of KnotStrand's and place them in the
        order of the knot (Find which termini are linked across gaps)
        """
        if not strands:
            raise ValueError("Must provide some strands to KnotStrandCollection")

        strand_ends: set[_StrandEnd] = set()
        for strand in strands:
            strand_ends.add(
                (
                    strand,
                    _LeftOrRight.LEFT,
                )
            )
            strand_ends.add((strand, _LeftOrRight.RIGHT))

        first = next(iter(strand_ends))
        strand_ends.remove(first)
        ordered_strand_ends: list[_StrandEnd] = [first]
        while strand_ends:
            previous_end = ordered_strand_ends[-1]
            candidates = _filter_candidate_ends(previous_end, strand_ends)
            if not candidates:
                raise ValueError(
                    f"Unpaired strand ends, remain, but unable to find match for {previous_end}"
                )

            next_strand_end = min(
                candidates,
                key=partial(_distance_to_candidate, previous_end=previous_end),
            )
            ordered_strand_ends.append(next_strand_end)
            strand_ends.remove(next_strand_end)

        return cls(strands=tuple(strands))

    def to_knot(self) -> Knot:
        raise NotImplementedError()


@attr.frozen
class Coord:
    row: int
    col: int

    @classmethod
    def new(cls, row: int, col: int) -> "Coord":
        return cls(row=row, col=col)

    @classmethod
    def from_tuple(cls, x_y: tuple[int, int]) -> "Coord":
        x, y = x_y
        return cls.new(x, y)

    @property
    def as_tuple(self) -> tuple[int, int]:
        return (self.row, self.col)

    def neighbors(self) -> set["Coord"]:
        return set(
            [
                Coord.new(self.row - 1, self.col - 1),
                Coord.new(self.row - 1, self.col),
                Coord.new(self.row - 1, self.col + 1),
                Coord.new(self.row, self.col - 1),
                Coord.new(self.row, self.col + 1),
                Coord.new(self.row + 1, self.col - 1),
                Coord.new(self.row + 1, self.col),
                Coord.new(self.row + 1, self.col + 1),
            ]
        )

    def cardinal_neighbors(self) -> set["Coord"]:
        return set(
            [
                Coord.new(self.row - 1, self.col),
                Coord.new(self.row, self.col - 1),
                Coord.new(self.row, self.col + 1),
                Coord.new(self.row + 1, self.col),
            ]
        )

    def diagonal_neighbors(self) -> set["Coord"]:
        return set(
            [
                Coord.new(self.row - 1, self.col - 1),
                Coord.new(self.row - 1, self.col + 1),
                Coord.new(self.row + 1, self.col - 1),
                Coord.new(self.row + 1, self.col + 1),
            ]
        )


def _closest_true_neighbor(
    subject: Coord,
    remaining_coords: set[Coord],
) -> Optional[Coord]:
    nearest_cardinal_neighbor = min(
        filter(remaining_coords.__contains__, subject.cardinal_neighbors()),
        key=ops.attrgetter("as_tuple"),
        default=None,
    )
    if nearest_cardinal_neighbor is not None:
        return nearest_cardinal_neighbor

    nearest_diagonal_neighbor = min(
        filter(remaining_coords.__contains__, subject.diagonal_neighbors()),
        key=ops.attrgetter("as_tuple"),
        default=None,
    )
    return nearest_diagonal_neighbor  # maybe None!


def _extract_single_strand(coords: set[Coord]) -> tuple[KnotStrand, set[Coord]]:
    """Take a set of coordinates that is true and return
    a single LineString vectorized from coords as long
    as the coordinates that were not included (the remainder).

    WARNING: coords is taken and returned in a mutated state!
    References to coords outside of this function will reflect that mutation
    """
    current: Coord = next(iter(coords))
    coords.remove(current)

    neighbor = _closest_true_neighbor(current, coords)
    if neighbor is None:
        return KnotStrand(strand=LineString([current.as_tuple])), coords
    coords.remove(neighbor)

    strand: deque[Coord] = deque([current, neighbor])
    while True:
        left = strand[0]
        next_: Optional[Coord] = _closest_true_neighbor(left, coords)
        if next_ is None:
            break

        coords.remove(next_)
        strand.appendleft(next_)

    while True:
        right = strand[-1]
        next_: Optional[Coord] = _closest_true_neighbor(right, coords)  # type: ignore[no-redef]
        if next_ is None:
            break

        coords.remove(next_)
        strand.append(next_)

    return (
        KnotStrand(strand=LineString(map(ops.attrgetter("as_tuple"), strand))),
        coords,
    )


def vectorize_thinned_mask(
    thinned_mask: npt.NDArray[np.bool_],
) -> KnotStrandCollection:
    strand_coords: set[Coord] = set(map(Coord.from_tuple, zip(*np.where(thinned_mask))))  # type: ignore[arg-type]

    knot_strands: list[KnotStrand] = []
    while strand_coords:
        new_strand, strand_coords = _extract_single_strand(strand_coords)
        knot_strands.append(new_strand)

    return KnotStrandCollection.new(knot_strands)
