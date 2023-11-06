from enum import Enum
from typing import Optional

import attr


class InvalidKnotError(ValueError):
    """Sequence of lines does not represent a valid knot"""


class OverUnder(Enum):
    OVER = 0
    UNDER = 1


@attr.frozen
class Crossing:
    id: str


@attr.frozen
class Line:
    """A line connecting to consecutive crossings
    (there are no crossing between them)
    """

    origin: Crossing
    origin_layer: OverUnder

    destination: Crossing
    destination_layer: OverUnder


@attr.frozen
class Knot:
    crossing_sequence: tuple[Line]

    @classmethod
    def new(cls, lines: tuple[Line]) -> "Knot":
        cls._validate_sequence(lines)
        return cls(crossing_sequence=lines)

    @staticmethod
    def _validate_sequence(lines: tuple[Line]) -> None:
        if not lines:
            # an empty sequence is the unknot
            return

        previous_destination: Optional[Crossing] = None
        for idx, line in enumerate(lines):
            if previous_destination is not None and previous_destination != line.origin:
                raise InvalidKnotError(
                    f"Sequence of lines contains inconsistent origin, destination information ({idx - 1} to {idx})"
                )
            previous_destination = line.destination

        start_node = lines[0].origin
        end_node = lines[-1].destination
        if start_node != end_node:
            raise InvalidKnotError(
                "A knot must start and end at the same crossing node"
            )
