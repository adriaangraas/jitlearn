import itertools
import warnings
from abc import ABC
from typing import Any, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from unet.unet import _SKIPS, unet
from unet.compose import Composable, Composition


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

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def attack_frame(self):
        return self._attack_frame

    def _extract_attack_frame(self, volume):
        if self._t_dim == 0:
            frame = volume[self._attack_frame, ...]
        elif self._t_dim == 1:
            frame = volume[:, self._attack_frame, ...]
        elif self._t_dim == 2:
            frame = volume[:, :, self._attack_frame, ...]
        else:
            raise NotImplementedError

        frame = torch.unsqueeze(frame, dim=self._t_dim)

        if volume.ndim == 5:  # 3D volumes: extract middle slice
            assert frame.shape[-1] % 2 == 1
            middle_frame = (frame.shape[-1] + 1) // 2 - 1
            frame = frame[..., middle_frame]
        return frame

    def crit(self, output, target):
        """Differentiable loss"""
        return self._crit(
            output,
            self._extract_attack_frame(target))

    def accuracy(self, output, target, accuracy_func) -> float:
        return accuracy_func(
            torch.squeeze(output, dim=self._t_dim),
            torch.squeeze(self._extract_attack_frame(target), dim=self._t_dim))

    def write_summary(self, summary_writer, batch):
        pass


class _Unet3DLevel(Composable):
    class Block(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int,
                     stride: tuple = (1, 1, 1)):
            super().__init__()
            self.append(nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False))
            self.append(nn.ReLU(inplace=True))

        def forward(self, x):
            x2 = self[0](x)
            x3 = self[1](x2)
            return x3

    def _up_block(self, in_channels, out_channels, stride,
                  input_shape_odd: Sequence) -> nn.Module:
        """Upsampling block"""

        # Output padding depends on input shape and stride, and is computed
        # as follows. If stride == 1 (in any direction), then output padding
        # is not necessary. If stride == 2, then output padding is necessary
        # when the input volume is even.
        # TODO: generalize this to higher strides?
        output_padding = tuple([1 if s == 2 and not odd else 0
                                for s, odd in zip(stride, input_shape_odd)])
        layers = []
        layers.append(nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            output_padding=output_padding,
            stride=stride,
            bias=False))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def __init__(self, *,
                 in_channels: int,
                 intermediate_channels: int,
                 out_channels: int,
                 stride=(1, 1, 1),
                 skip: str = None,
                 superskip: float = 0.,
                 input_shape_odd=(False, False, False),
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

        if skip is not None and stride == (1, 1, 1):
            raise ValueError(
                "Skip connection without downsampling is dubious?")

        self.skip = skip
        assert skip in _SKIPS
        self.superskip = superskip

        self._intermediate_channels = intermediate_channels

        # modules:
        self.pre_block = nn.Sequential(
            self.Block(in_channels, intermediate_channels),
            self.Block(intermediate_channels, intermediate_channels)
        )
        self.im_enc = self.Block(
            intermediate_channels,
            self.composed_module._in_channels if self.composed_module else intermediate_channels,
            stride=stride)
        # here deeper level is then called
        self.im_dec = self._up_block(
            self.composed_module._out_channels if self.composed_module else intermediate_channels,
            intermediate_channels,
            stride=stride,
            input_shape_odd=input_shape_odd)
        self.post_block = nn.Sequential(
            self.Block(
                intermediate_channels * (2 if skip == 'concat' else 1),
                intermediate_channels),
            self.Block(
                intermediate_channels,
                out_channels))

        if not isinstance(self.superskip, str):
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
        s = f"{self.__class__.__name__}: " \
            f"{self.in_channels}, {self._intermediate_channels}, {self.out_channels}"
        if self.skip is not None:
            s += f"+{self.skip}skip"
        return s


class Unet3d(SpatioTemporalModel):
    def __init__(self,
                 *args,
                 sequence_length: int,
                 attack_frame: int,
                 nr_levels: int = 3,
                 space_dims: int = 3,
                 with_temporal_downsampling: bool = True,
                 with_spatial_downsampling: bool = True,
                 initial_down_channels: int = 32,
                 skip='concat',
                 verbose=False,
                 superskip: float = 0.,
                 **kwargs):
        # t_dim=1 because [B,C,>>T<<,X,Y,Z] and B gets removed in forward()
        super().__init__(*args,
                         sequence_length=sequence_length,
                         attack_frame=attack_frame,
                         t_dim=1,
                         **kwargs)
        self._nr_levels = nr_levels
        self._with_temporal_downsampling = with_temporal_downsampling
        self._with_spatial_downsampling = with_spatial_downsampling
        self._initial_down_channels = initial_down_channels
        self._skip = skip
        self._superskip = superskip
        self._time_dims = 1 if sequence_length > 1 else 0
        self._space_dims = space_dims
        assert self._space_dims in (2, 3)
        self._nr_dims = self._time_dims + self._space_dims
        assert self._nr_dims in (2, 3, 4)

        # not abusing the channel dimension in these unets
        if self._nr_dims == 2:
            self._unet = spato_unet_3d(
                nr_levels=nr_levels,
                downsampling=(with_spatial_downsampling,
                              with_spatial_downsampling,
                              False),
                input_shape_odd=(False, False, True),  # by convention
                initial_down_channels=initial_down_channels,
                verbose=True,
                skip=skip,
                superskip=superskip)
        elif self._nr_dims == 3:
            if self._time_dims == 1:
                self._unet = spato_unet_3d(
                    nr_levels=nr_levels,
                    downsampling=(with_temporal_downsampling,
                                  with_spatial_downsampling,
                                  with_spatial_downsampling),
                    input_shape_odd=(True, False, False),  # by convention
                    initial_down_channels=initial_down_channels,
                    verbose=True,
                    skip=skip,
                    superskip=superskip)
            elif self._time_dims == 0:
                self._unet = spato_unet_3d(
                    nr_levels=nr_levels,
                    downsampling=tuple([with_spatial_downsampling] * 3),
                    input_shape_odd=(False, False, True),  # by convention
                    initial_down_channels=initial_down_channels,
                    verbose=True,
                    skip=skip,
                    superskip=superskip)
        else:
            # if True:
            self._unet = unet_nd(
                num_dims=4,
                nr_levels=nr_levels,
                downsampling=(with_temporal_downsampling,
                              with_spatial_downsampling,
                              with_spatial_downsampling,
                              with_spatial_downsampling),
                initial_down_channels=initial_down_channels,
                verbose=verbose,
                skip=skip,
                superskip=superskip)

        self._crit = nn.MSELoss()

    def forward(self, x, *args, **kwargs):
        x = torch.unsqueeze(x, dim=1)  # add channel dimension
        assert x.ndim == 6, "x should have {} dimensions, but has {}".format(
            6, x.ndim)

        if self._nr_dims == 2:
            x = torch.squeeze(x, dim=2)
        elif self._nr_dims == 3:
            if self._time_dims == 0:
                x = torch.squeeze(x, dim=2)

            if self._space_dims == 2:
                x = torch.squeeze(x, dim=-1)

        x = self._unet(x, *args, **kwargs)

        if self._nr_dims == 2:
            x = torch.unsqueeze(x, dim=2)
        if self._nr_dims == 3:
            if self._space_dims == 2:
                x = torch.unsqueeze(x, dim=-1)

            if self._time_dims == 0:
                x = torch.unsqueeze(x, dim=2)

        x = torch.squeeze(x, dim=1)  # remove channel dimension
        assert x.ndim == 5
        # the 3D U-Net outputs a full spatiotemporal volume and the
        # SpatioTemporalModel defines a 1-frame, 1-slice output
        x = self._extract_attack_frame(x)
        return x

    def get_config(self) -> dict:
        return {'sequence_length': self.sequence_length,
                'space_dims': self._space_dims,
                'attack_frame': self.attack_frame,
                'nr_levels': self._nr_levels,
                'with_temporal_downsampling': self._with_temporal_downsampling,
                'with_spatial_downsampling': self._with_spatial_downsampling,
                'skip': self._skip,
                'initial_down_channels': self._initial_down_channels,
                'superskip': self._superskip}

    def __str__(self):
        return str(self._unet)

    def write_summary(self, summary_writer, batch):
        if hasattr(self._unet._composables[0], 'alpha'):
            summary_writer.add_scalar(
                '/Model/superskip',
                self._unet._composables[0].alpha,
                batch)


def spato_unet_3d(in_channels: int = 1,
                  out_channels: int = 1,
                  nr_levels: int = 3,
                  initial_down_channels=32,
                  downsampling=(True, True, True),
                  input_shape_odd=(False, False, False),
                  verbose=False,
                  skip='concat',
                  **kwargs):
    """Proper 3D U-Net that treats the first dimension as time."""
    assert skip in [None, 'concat', 'residual']

    stride = tuple([2 if d else 1 for d in downsampling])
    strides = [stride] * (nr_levels - 1) + [(1, 1, 1)]
    inter_ch = [initial_down_channels * 2 ** l for l in range(nr_levels)]
    if skip is not None:
        skips = [skip if l != nr_levels - 1 else None
                 for l in range(nr_levels)]  # all but bottomlvl
    else:
        skips = [None] * nr_levels

    return unet(
        _Unet3DLevel,
        in_channels,
        out_channels,
        intermediate_channels=inter_ch,
        strides=strides,
        skips=skips,
        input_shape_odd=input_shape_odd,
        verbose=verbose,
        **kwargs)