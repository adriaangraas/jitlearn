import itertools
import warnings
from typing import Any, Sequence, Union

import torch.nn as nn

class Composable(nn.Module):
    """Composable
    In principle you can see a U-Net as
        PSI(x) = f(g(...(h(x), x), x), x)

    The functions f, g, etc. are custom but typically only with different
    spatial resolution, i.e.
        PSI(x) = f_1(f_2(f_3(x), x, x, x).

    The nn.Module structure that we need to formalize is this
        f with an __init__(g) and forward(x) that guarantees
        g to be called, and the output value to be conformant (except for
        the deepest level).

    This composable type should be subclassed.
    """

    def __init__(self, in_channels, out_channels, composed_module=None):
        super().__init__()
        self.composed_module = composed_module
        self.in_channels = in_channels
        self.out_channels = out_channels

    @property
    def composed_module(self):
        return self._composed_with

    @composed_module.setter
    def composed_module(self, value):
        self._composed_with = value

    def compose(self, input, **kwargs):
        if self._composed_with is not None:
            return self.composed_module(input, **kwargs)

        return input


class Composition(nn.Module):
    @staticmethod
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def __init__(self, *composables: Composable):
        super().__init__()
        self._composables = composables

        for l, m in self.pairwise(composables):
            l.composed_module = m  # chain l to m

        self._entry = composables[0]

    def forward(self, *input: Any, **kwargs: Any):
        return self._entry(*input, **kwargs)

    def __str__(self):
        return '; \n'.join((str(c) for c in self._composables))
