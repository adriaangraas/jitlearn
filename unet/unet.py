from abc import ABC
from typing import Any, Sequence, Union

import numpy as np
from torch import nn

from unet.compose import Composable, Composition

_SKIPS = [None, 'concat', 'residual']


class SpatioTemporalModel(nn.Module, ABC):
    """Model that handles sequence data."""

    def __init__(self, *args,
                 sequence_length: int,
                 attack_frame: int,
                 t_dim: int,
                 **kwargs):
        assert sequence_length > 0
        assert attack_frame >= 0
        super().__init__(*args, **kwargs)
        self._sequence_length = sequence_length
        self._attack_frame = attack_frame
        self._t_dim = t_dim


def unet(
    lvl_factory,
    in_channels: int,
    out_channels: int,
    intermediate_channels: Sequence,
    strides: Sequence,
    skips: Sequence,
    verbose: bool = False,
    **kwargs):
    """U-net with custom options for strides, channels, etc."""
    assert len(intermediate_channels) == len(strides) == len(skips)
    nr_levels = len(intermediate_channels)

    if not np.all(np.array(strides[-1]) == 1) or skips[-1] != None:
        raise ValueError("Bottom level should not use skips or downsample.")

    def _lvl(in_channels, intermediate_channels, out_channels, stride,
             skip, **kwargs):
        return lvl_factory(
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            out_channels=out_channels,
            stride=stride,
            skip=skip,
            # input_shape_odd=input_shape_odd,
            **kwargs
        )

    levels = []
    for i in range(nr_levels):
        levels.append(
            _lvl(in_channels if i == 0 else intermediate_channels[i - 1],
                 intermediate_channels[i],
                 out_channels if i == 0 else intermediate_channels[i - 1],
                 strides[i],
                 skips[i],
                 # input_shape_odd
                 **kwargs
                 ))

    net = Composition(*levels)
    if verbose:
        print(net)

    return net