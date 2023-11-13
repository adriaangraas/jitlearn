from typing import Any
import torch
import torch.nn as nn
from unet.unet import _SKIPS, unet
from unet.compose import Composable


class _Unet2DLevel(Composable):
    def _block(self, in_channels: int, out_channels: int,
               stride: tuple = (1, 1)) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels, stride) -> nn.Module:
        """Upsampling block"""
        if stride == (1, 1):
            output_padding = 0
        elif stride == (2, 2):
            output_padding = 1
        else:
            raise NotImplementedError()  # TODO: formalize

        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                output_padding=output_padding,
                stride=stride,
                bias=False,
            ),
            nn.ReLU(inplace=True)
        )

    def __init__(self, *,
                 in_channels: int,
                 intermediate_channels: int,
                 out_channels: int,
                 stride=(1, 1),
                 skip: str = None,
                 composed_module: Composable = None,
                 ):
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

        if skip is not None and stride == (1, 1):
            raise ValueError(
                "Skip connection without downsampling is dubious?")

        self.skip = skip
        assert skip in _SKIPS

        self._intermediate_channels = intermediate_channels

        # modules:
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
        self.im_dec = self._up_block(
            self.composed_module._out_channels if self.composed_module else intermediate_channels,
            intermediate_channels,
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

    def forward(self, input: Any, **kwargs: Any):
        # 1. convolution layer
        x1 = self.pre_block(input, **kwargs)

        # 2. deeper level call
        x2 = self.im_enc(x1, **kwargs)  # downsampling
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
        return y

    def __str__(self):
        s = f"in,inter,out:{self.in_channels},{self._intermediate_channels},{self.out_channels}"
        if self.skip is not None:
            s += f"+{self.skip}skip"
        return s


def unet_2d(in_channels: int, out_channels: int = 1,
            nr_levels: int = 3,
            initial_down_channels=32,
            verbose=False,
            skip='concat',
            **kwargs):
    """2D U-Net that halves the resolution and doubles the channels in
    every level."""
    assert skip in _SKIPS

    strides = [(2, 2)] * (nr_levels - 1) + [(1, 1)]
    inter_ch = [initial_down_channels * 2 ** l for l in range(nr_levels)]
    if skip is not None:
        skips = [skip if l != nr_levels - 1 else None
                 for l in range(nr_levels)]  # all but bottomlvl
    else:
        skips = [None] * nr_levels

    kwargs_defaults = {
        'intermediate_channels': inter_ch,
        'strides': strides,
        'skips': skips,
        'verbose': verbose,
    }
    kwargs_defaults.update(kwargs)
    return unet(_Unet2DLevel, in_channels, out_channels, **kwargs_defaults)
