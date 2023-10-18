from enum import Enum

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
    def _validate_sequence(lines: tuple[Line]):
        if not lines:
            # an empty sequence is the unknot
            return

        start_node = lines[0].origin
        end_node = lines[-1].destination
        if start_node != end_node:
            raise InvalidKnotError(
                "A knot must start and end at the same crossing node"
            )
