from pathlib import Path

import attr
from attr import field
from attr.validators import instance_of
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt


@attr.frozen
class RgbRaster:
    data: npt.NDArray[np.int8] = field(validator=instance_of(np.ndarray))

    @data.validator
    def has_correct_shape(self, attribute, value):  # type: ignore[no-untyped-def]
        array_shape = value.shape
        n_dimensions = len(array_shape)
        if not n_dimensions == 3:
            raise ValueError(
                f"RGB array expected to be 3D (x, y, bands); got {n_dimensions}"
            )

        n_channels = value.shape[2]
        if n_channels not in (3, 4):
            raise ValueError(
                f"RGB array expected to have 3 or 4 channels; got {n_channels}"
            )

        for channel in range(2):
            channel_values = value[:, :, channel]
            min_val = np.min(channel_values)
            max_val = np.max(channel_values)
            if min_val < 0 or max_val > 255:
                raise ValueError(
                    f"RGB channel values must be integers in [0, 255]; got min: {min_val}, max: {max_val} in channel {channel}"
                )

    @classmethod
    def from_png(cls, path: Path) -> "RgbRaster":
        data = iio.imread(path)
        return cls(data=data)

    @property
    def red_channel(self) -> npt.NDArray[np.int8]:
        return self.data[:, :, 0]

    @property
    def green_channel(self) -> npt.NDArray[np.int8]:
        return self.data[:, :, 1]

    @property
    def blue_channel(self) -> npt.NDArray[np.int8]:
        return self.data[:, :, 2]


@attr.frozen
class GrayscaleRaster:
    data: npt.NDArray[np.int8] = field(validator=instance_of(np.ndarray))

    @data.validator
    def has_correct_shape(self, attribute, value):  # type: ignore[no-untyped-def]
        array_shape = value.shape
        n_dimensions = len(array_shape)
        if not n_dimensions == 2:
            raise ValueError(
                f"Grayscale array expected to be 2D (x, y); got {n_dimensions}"
            )

    @property
    def values(self) -> npt.NDArray[np.int8]:
        return self.data


@attr.frozen
class BinaryMask:
    data: npt.NDArray[np.bool_] = field(validator=instance_of(np.ndarray))

    @data.validator
    def has_correct_shape(self, attribute, value):  # type: ignore[no-untyped-def]
        array_shape = value.shape
        n_dimensions = len(array_shape)
        if not n_dimensions == 2:
            raise ValueError(
                f"Grayscale array expected to be 2D (x, y); got {n_dimensions}"
            )

    @property
    def values(self) -> npt.NDArray[np.bool_]:
        return self.data


def rgb_to_grayscale(rgb: RgbRaster) -> GrayscaleRaster:
    """Calculate luminance from RGB channels:
    https://en.wikipedia.org/wiki/Grayscale
    """
    luminance_data = (
        0.2126 * rgb.red_channel
        + 0.7152 * rgb.green_channel
        + 0.0722 * rgb.blue_channel
    )
    return GrayscaleRaster(data=luminance_data)


def grayscale_to_binary_mask(
    grayscale: GrayscaleRaster,
    threshold_quantile: float,
    large_values_are_true: bool = True,
) -> BinaryMask:
    """Create a binary mask from a grayscale image.
    Order the pixels and calculate percentiles.
    Partition pixels into those less than the
    and those greater than or equal to the threshold
    percentile.
    """
    if threshold_quantile < 0 or threshold_quantile > 1:
        raise ValueError("Threshold quantile must be between 0 and 1 inclusive")

    if large_values_are_true:
        quantile = np.quantile(grayscale.values.ravel(), threshold_quantile)
        mask = grayscale.values >= quantile
    else:
        threshold_quantile = 1 - threshold_quantile
        quantile = np.quantile(grayscale.values.ravel(), threshold_quantile)
        mask = grayscale.values <= quantile

    return BinaryMask(data=mask)


def read_raster_knot(path: Path) -> BinaryMask:
    rgb_raster = RgbRaster.from_png(path)
    grayscale_raster = rgb_to_grayscale(rgb_raster)
    binary_mask = grayscale_to_binary_mask(
        grayscale_raster, 0.9, large_values_are_true=False
    )
    return binary_mask
