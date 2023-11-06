from collections import deque
import operator as ops
from typing import Optional

import attr
import numpy as np
import numpy.typing as npt
from attr.validators import instance_of
from shapely.geometry import Point, LineString


@attr.frozen
class KnotStrand:
    strand: LineString = attr.ib(validator=instance_of(LineString))

    @strand.validator
    def is_nonempty(self, attribute, value):  # type: ignore[no-untyped-def]
        if len(value.coords) == 0:
            raise ValueError("Knot strands must be non-empty")

    @property
    def terminus_1(self) -> Point:
        return Point(self.strand.coords[0])

    @property
    def terminus_2(self) -> Point:
        return Point(self.strand.coords[-1])

    @property
    def ends(self) -> tuple[Point, Point]:
        return (
            self.terminus_1,
            self.terminus_2,
        )


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
) -> tuple[KnotStrand, ...]:
    strand_coords: set[Coord] = set(map(Coord.from_tuple, zip(*np.where(thinned_mask))))  # type: ignore[arg-type]

    knot_strands: list[KnotStrand] = []
    while strand_coords:
        new_strand, strand_coords = _extract_single_strand(strand_coords)
        knot_strands.append(new_strand)

    return tuple(knot_strands)
