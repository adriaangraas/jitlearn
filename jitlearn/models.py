from abc import ABC, abstractmethod
import torch
import torch.nn as nn


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
        # not sure how to do this ..., this yields some device assert errors
        # idx = torch.IntTensor(self._attack_frame).to(volume.get_device())
        # frame = torch.index_select(volume,
        #                            dim=self._t_dim,
        #                            index=idx)
        # frame = torch.gather(volume, self._t_dim, idx)

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
            self._extract_attack_frame(target)
        )

    def accuracy(self, output, target, accuracy_func) -> float:
        return accuracy_func(
            torch.squeeze(output, dim=self._t_dim),
            torch.squeeze(self._extract_attack_frame(target), dim=self._t_dim))

    def write_summary(self, summary_writer, batch):
        pass