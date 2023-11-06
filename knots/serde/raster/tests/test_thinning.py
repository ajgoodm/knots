# pylint: disable=redefined-builtin
from pathlib import Path

import numpy as np
import pytest

from knots.serde.raster.formats import read_raster_knot
from knots.serde.raster.thinning import (
    n_transitions,
    n_true_neighbors,
    zhang_suen_thinning_algorithm,
)


TREFOIL_PATH = Path(__file__).parent / "data" / "trefoil.png"


def _input_to_bool(char: str) -> np.bool_:
    if char == " ":
        return np.bool_(False)
    if char == "*":
        return np.bool_(True)
    raise ValueError(f"unexpected input character {char}")


def _output_to_int(char: str) -> np.int8:
    if char == " ":
        return np.int8(-1)
    if str.isnumeric(char):
        return np.int8(int(char))
    raise ValueError(f"unexpected output character {char}")


class TestZhangSuen:
    # fmt: off
    @pytest.mark.parametrize(
        ("input", "exp_result"),
        [
            (
                [
                    "     ",
                    " * * ",
                    "  *  ",
                    " * * ",
                    "     ",
                ],
                [
                    "     ",
                    " 131 ",
                    " 343 ",
                    " 131 ",
                    "     ",
                ]
            ),
            (
                [
                    "* * ",
                    " ** ",
                    "   *",
                    " ** ",
                ],
                [
                    "    ",
                    " 23 ",
                    " 23 ",
                    "    ",
                ]
            )
        ],
    )
    def test_n_transitions(
        self,
        input,
        exp_result,
    ):
        input_: list[list[np.bool_]] = []
        for row in input:
            input_.append(list(map(_input_to_bool, row)))
        input_ = np.array(input_)

        exp_result_: list[list[np.int8]] = []
        for row in exp_result:
            exp_result_.append(list(map(_output_to_int, row)))
        exp_result_ = np.array(exp_result_)

        obs_result = n_transitions(input_)
        assert np.all(obs_result == exp_result_)

    @pytest.mark.parametrize(
        ("input", "exp_result"),
        [
            (
                [
                    "     ",
                    " * * ",
                    "  *  ",
                    " * * ",
                    "     ",
                ],
                [
                    "     ",
                    " 131 ",
                    " 343 ",
                    " 131 ",
                    "     ",
                ]
            ),
            (
                [
                    "* * ",
                    " ** ",
                    "   *",
                    " ** ",
                ],
                [
                    "    ",
                    " 33 ",
                    " 45 ",
                    "    ",
                ]
            )
        ],
    )
    def test_n_true_neighbors(self, input, exp_result):
        input_: list[list[np.bool_]] = []
        for row in input:
            input_.append(list(map(_input_to_bool, row)))
        input_ = np.array(input_)

        exp_result_: list[list[np.int8]] = []
        for row in exp_result:
            exp_result_.append(list(map(_output_to_int, row)))
        exp_result_ = np.array(exp_result_)

        obs_result = n_true_neighbors(input_)
        assert np.all(obs_result == exp_result_)
    # fmt: on

    def test_zhang_suen_thinning_algorithm(self):
        mask = read_raster_knot(TREFOIL_PATH)
        zhang_suen_thinning_algorithm(mask.data)
