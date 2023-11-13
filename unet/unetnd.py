from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from unet.compose import Composable, Composition
from unet.convnd.convnd import ConvNd


class _UnetNdLevel(Composable):
    def _block(self, in_channels: int, out_channels: int,
               transpose: bool = False,
               stride: tuple = (1, 1, 1, 1),
               output_padding=(0, 1, 1, 0),
               ) -> nn.Module:
        return nn.Sequential(
            ConvNd(
                in_channels=in_channels,
                out_channels=out_channels,
                num_dims=self._num_dims,
                kernel_size=(3, 3, 3, 3),
                padding=1,
                output_padding=output_padding,
                is_transposed=transpose,
                stride=stride,
                use_bias=False),
            nn.ReLU(inplace=True)
        )

    def __init__(self, *,
                 num_dims: int,
                 in_channels: int,
                 intermediate_channels: int,
                 out_channels: int,
                 stride=(1, 1, 1),
                 skip: str = None,
                 superskip: float = 0.,
                 composed_module: Composable = None,
                 with_batchnorm=False,
                 signal_length=0):
        """
        down_channels at level i must equal in_channels at level i+1

        :param in_channels:
        :param intermediate_channels: The number of channels it will pass on the next
        level, or number of channels that it will use in-between input and output.
        :param out_channels:
        :param skip: None, 'concat' or 'residual'
        :param composed_module:
        :param with_temporal_downsampling:
        :param with_spatial_downsampling:
        """
        super().__init__(in_channels, out_channels, composed_module)
        self._num_dims = num_dims

        assert not with_batchnorm, "Not implemented."
        assert signal_length != 0, "Not implemented."

        if skip is not None and stride == (1, 1, 1):
            raise ValueError(
                "Skip connection without downsampling is dubious?")

        self.skip = skip
        assert skip in [None, 'residual', 'concat']
        self.superskip = superskip

        self._intermediate_channels = intermediate_channels

        self.pre_block = nn.Sequential(
            self._block(in_channels,
                        intermediate_channels),
            self._block(intermediate_channels,
                        intermediate_channels)
        )
        self.im_enc = self._block(
            intermediate_channels,
            self.composed_module._in_channels if self.composed_module else intermediate_channels,
            stride=stride)
        # here deeper level is then called
        self.im_dec = self._block(
            self.composed_module._out_channels if self.composed_module else intermediate_channels,
            intermediate_channels,
            transpose=True,
            stride=stride)
        self.post_block = nn.Sequential(
            self._block(
                intermediate_channels * (2 if skip == 'concat' else 1),
                intermediate_channels,
            ),
            self._block(
                intermediate_channels,
                out_channels,
            ))

        self.alpha = nn.Parameter(torch.ones(1) * superskip)

    def forward(self, input):
        # 1. convolution layer
        x1 = self.pre_block(input)

        # 2. deeper level call
        x2 = self.im_enc(x1)  # downsampling
        x2 = self.compose(x2)  # feeding into the next level
        x1_up = self.im_dec(x2)  # upsampling

        # 3. concatenation with skip/residual connection
        if self.skip is None:
            x1 = x1_up
        elif self.skip == 'concat':
            x1 = torch.cat([x1, x1_up], dim=1)
        elif self.skip == 'residual':
            x1 = x1 + x1_up
        else:
            raise ValueError(f"Unknown value '{self.skip}' for `skip`.")

        # 4. feature merging
        y = self.post_block(x1)
        if self.superskip == 'residual':
            y = y + input
        else:
            y = self.alpha * input + (1 - self.alpha) * y
        return y

    def __str__(self):
        s = f"in,inter,out:{self.in_channels},{self._intermediate_channels},{self.out_channels}"
        if self.skip is not None:
            s += f"+{self.skip}skip"
        return s


def unet_nd(
    num_dims: int = 4,
    in_channels: int = 1,
    out_channels: int = 1,
    nr_levels: int = 3,
    initial_down_channels: int = 32,
    downsampling: Sequence = None,
    verbose: bool = False,
    skip: str = 'concat',
    **kwargs):
    assert skip in [None, 'concat', 'residual']
    assert len(downsampling) == num_dims
    stride = tuple([2 if b else 1 for b in downsampling])
    strides = [stride] * (nr_levels - 1) + [(1,) * num_dims]
    intermediate_channels = [initial_down_channels * 2 ** l for l in
                             range(nr_levels)]
    if skip is not None:
        skips = [skip if l != nr_levels - 1 else None
                 for l in range(nr_levels)]  # all but bottomlvl
    else:
        skips = [None] * nr_levels

    assert len(intermediate_channels) == len(strides) == len(skips)
    nr_levels = len(intermediate_channels)

    if not np.all(np.array(strides[-1]) == 1) or skips[-1] != None:
        raise ValueError(
            "Bottom level should not use skips or downsample.")

    def _lvl(in_channels, intermediate_channels, out_channels, stride,
             skip):
        return _UnetNdLevel(
            num_dims=num_dims,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            out_channels=out_channels,
            stride=stride,
            skip=skip,
            **kwargs)

    levels = []
    for i in range(nr_levels):
        levels.append(
            _lvl(in_channels if i == 0 else intermediate_channels[i - 1],
                 intermediate_channels[i],
                 out_channels if i == 0 else intermediate_channels[i - 1],
                 strides[i],
                 skips[i]))

    net = Composition(*levels)
    if verbose:
        print(net)
    return net