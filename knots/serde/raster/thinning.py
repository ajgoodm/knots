from typing import TypeVar

import numpy as np
import numpy.typing as npt


def _validate_input(bool_raster: npt.NDArray[np.bool_]) -> None:
    if len(bool_raster.shape) != 2:
        raise ValueError("We work with 2D rasters here!")
    if any(dim < 3 for dim in bool_raster.shape):
        raise ValueError(
            "We only perform thinning for 2D rasters that are at least 3 wide in all dimensions"
        )


def n_transitions(bool_raster: npt.NDArray[np.bool_]) -> npt.NDArray[np.int8]:
    """Attempts to count the number of False -> True transitions
    for the pixels surrounding a subject pixel. For example:

        * . .
        * S *       * - True
        . * .       . - False

    For the subject pixel S, traversing the surrounding pixels (clockwise)
    and counting the number of transitions yields 3! We do not calculate
    values for edge pixels and insert a sentinel value (-1) on the edges.

    For the above example, we would return

        -1, -1, -1
        -1,  3, -1
        -1, -1, -1
    """
    _validate_input(bool_raster)
    output = np.full(bool_raster.shape, -1, dtype=np.int8)
    output[1:-1, 1:-1] = np.full(
        (bool_raster.shape[0] - 2, bool_raster.shape[1] - 2), 0, dtype=np.int8
    )

    east = bool_raster[:, 1:]
    west = bool_raster[:, :-1]

    # fmt: off
    transition_occurred = np.logical_and(np.logical_not(east), west).astype(np.int8)
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[:-2, :-1]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[:-2, 1:]

    transition_occurred = np.logical_and(np.logical_not(west), east).astype(np.int8)
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[2:, :-1]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[2:, 1:]

    north = bool_raster[:-1, :]
    south = bool_raster[1:, :]

    transition_occurred = np.logical_and(np.logical_not(north), south).astype(np.int8)
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[:-1, :-2]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[1:, :-2]

    transition_occurred = np.logical_and(np.logical_not(south), north).astype(np.int8)
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[:-1, 2:]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + transition_occurred[1:, 2:]
    # fmt: on

    return output


def n_true_neighbors(bool_raster: npt.NDArray[np.bool_]) -> npt.NDArray[np.int8]:
    """For a given pixel, count the number of neighboring pixels
    that are True. For example:

        * . .
        * S *       * - True
        . * .       . - False

    For the subject pixel S, traversing the surrounding pixels
    and counting the True pixels yields 4! We do not calculate
    values for edge pixels and insert a sentinel value (-1) on the edges.

    For the above example, we would return

        -1, -1, -1
        -1,  4, -1
        -1, -1, -1
    """
    _validate_input(bool_raster)
    output = np.full(bool_raster.shape, -1, dtype=np.int8)
    output[1:-1, 1:-1] = np.full(
        (bool_raster.shape[0] - 2, bool_raster.shape[1] - 2), 0, dtype=np.int8
    )

    input_as_int = bool_raster.astype(np.int8)
    # fmt: off
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[:-2, :-2]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[:-2, 1:-1]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[:-2, 2:]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[1:-1, :-2]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[1:-1, 2:]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[2:, :-2]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[2:, 1:-1]
    output[1:-1, 1:-1] = output[1:-1, 1:-1] + input_as_int[2:, 2:]
    # fmt: on

    return output


RasterElement = TypeVar("RasterElement", bound=np.generic)


def _north(raster: npt.NDArray[RasterElement]) -> npt.NDArray[RasterElement]:
    return raster[:-2, 1:-1]


def _east(raster: npt.NDArray[RasterElement]) -> npt.NDArray[RasterElement]:
    return raster[1:-1, 2:]


def _south(raster: npt.NDArray[RasterElement]) -> npt.NDArray[RasterElement]:
    return raster[2:, 1:-1]


def _west(raster: npt.NDArray[RasterElement]) -> npt.NDArray[RasterElement]:
    return raster[1:-1, :-2]


def _any_north_east_south_is_false(
    bool_raster: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """For a given pixel, the pixel is True iff its
    north, east, or south neighbor is False

    All exterior pixels are marked false and not
    considered in the thinning algorithm.
    """
    _validate_input(bool_raster)

    output = np.full(bool_raster.shape, np.bool_(False), dtype=np.bool_)
    output[1:-1, 1:-1] = np.logical_or(
        np.logical_or(
            np.logical_not(_north(bool_raster)), np.logical_not(_east(bool_raster))
        ),
        np.logical_not(_south(bool_raster)),
    )
    return output


def _any_east_south_west_is_false(
    bool_raster: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """For a given pixel, the pixel is True iff its
    east, south, or west neighbor is False

    All exterior pixels are marked false and not
    considered in the thinning algorithm.
    """
    _validate_input(bool_raster)

    output = np.full(bool_raster.shape, np.bool_(False), dtype=np.bool_)
    output[1:-1, 1:-1] = np.logical_or(
        np.logical_or(
            np.logical_not(_east(bool_raster)), np.logical_not(_south(bool_raster))
        ),
        np.logical_not(_west(bool_raster)),
    )
    return output


def _any_west_north_east_is_false(
    bool_raster: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """For a given pixel, the pixel is True iff its
    west, north, east neighbor is False

    All exterior pixels are marked false and not
    considered in the thinning algorithm.
    """
    _validate_input(bool_raster)

    output = np.full(bool_raster.shape, np.bool_(False), dtype=np.bool_)
    output[1:-1, 1:-1] = np.logical_or(
        np.logical_or(
            np.logical_not(_west(bool_raster)), np.logical_not(_north(bool_raster))
        ),
        np.logical_not(_east(bool_raster)),
    )
    return output


def _any_south_west_north_is_false(
    bool_raster: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """For a given pixel, the pixel is True iff its
    south, west, or north neighbor is False

    All exterior pixels are marked false and not
    considered in the thinning algorithm.
    """
    _validate_input(bool_raster)

    output = np.full(bool_raster.shape, np.bool_(False), dtype=np.bool_)
    output[1:-1, 1:-1] = np.logical_or(
        np.logical_or(
            np.logical_not(_south(bool_raster)), np.logical_not(_west(bool_raster))
        ),
        np.logical_not(_north(bool_raster)),
    )
    return output


def zhang_suen_thinning_algorithm(
    bool_raster: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Thin a boolean raster image to a 1-pixel wide skeleton image."""

    output = bool_raster.copy()

    while True:
        n_transitions_ = n_transitions(output)
        n_transitions_eligible = n_transitions_ == 1

        n_true_neighbors_ = n_true_neighbors(output)
        n_true_neighbors_eligible = np.logical_and(
            n_true_neighbors_ >= 2,
            n_true_neighbors_ <= 6,
        )

        pixels_were_removed = False
        to_remove = np.logical_and(
            output,
            np.logical_and(
                n_transitions_eligible,
                np.logical_and(
                    n_true_neighbors_eligible,
                    np.logical_and(
                        _any_north_east_south_is_false(output),
                        _any_east_south_west_is_false(output),
                    ),
                ),
            ),
        )
        output = np.logical_and(output, np.logical_not(to_remove))
        if np.any(to_remove):
            pixels_were_removed = True

        if pixels_were_removed:
            # The previous pass changed our raster, so we must
            # recalculate our heuristics.
            n_transitions_ = n_transitions(output)
            n_transitions_eligible = n_transitions_ == 1

            n_true_neighbors_ = n_true_neighbors(output)
            n_true_neighbors_eligible = np.logical_and(
                n_true_neighbors_ >= 2,
                n_true_neighbors_ <= 6,
            )

        to_remove = np.logical_and(
            output,
            np.logical_and(
                n_transitions_eligible,
                np.logical_and(
                    n_true_neighbors_eligible,
                    np.logical_and(
                        _any_west_north_east_is_false(output),
                        _any_south_west_north_is_false(output),
                    ),
                ),
            ),
        )
        output = np.logical_and(output, np.logical_not(to_remove))
        if np.any(to_remove):
            pixels_were_removed = True

        if not pixels_were_removed:
            break

    return output
