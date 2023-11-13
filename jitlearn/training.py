import os
import itertools
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.functional import mse_loss
import warnings

from tqdm import tqdm


class Accuracy(ABC):
    def __init__(self, roi: tuple = None, display_name: str = None):
        self._roi = roi
        self._display_name = display_name

    @property
    def roi(self) -> tuple:
        return self._roi

    @abstractmethod
    def __call__(self, output, target):
        pass

    def _display_name_with_roi(self, name: str) -> str:
        if self._display_name is not None:
            return self._display_name
        else:
            if self.roi is not None:
                return f"{name} {self.roi}"

        return name

    @abstractmethod
    def __str__(self):
        pass


class RMSE(Accuracy):
    @staticmethod
    def _crop(img, top, left, height, width):
        assert img.shape[-2] > top + height
        assert img.shape[-1] > left + width
        return img[..., top:top + height, left:left + width]

    def __init__(self, roi: tuple = None, display_name: str = None):
        super().__init__(roi=roi, display_name=display_name)

    def __call__(self, output, target):
        if self._roi is not None:
            output = self._crop(output, *self._roi)
            target = self._crop(target, *self._roi)

        return self.rmse(output, target)

    @staticmethod
    def rmse(output, target):
        output = output.detach()
        target = target.detach()
        return mse_loss(output, target).cpu().numpy().item()

    def __str__(self):
        return self._display_name_with_roi("RMSE")


class AccuracyAggregate:
    """Helper to get the total accuracy of multiple batches."""

    def __init__(self, accuracy):
        super().__init__()
        self.accuracy = accuracy
        self.items = []
        self.batch_sizes = []
        self._means = []

    @property
    def total(self) -> int:
        return sum(self.items)

    @property
    def nr_items(self) -> float:
        return sum(self.batch_sizes)

    def __call__(self, output, target) -> float:
        s = sum([self.accuracy(output[b], target[b])  # sum over batches
                 for b in range(output.shape[0])])
        self.items.append(s)
        self.batch_sizes.append(output.shape[0])
        mean = self.average()
        self._means.append(mean)
        return mean

    def average(self, end=None) -> float:
        if end is None:
            return self.total / self.nr_items
        else:
            return np.average(self.items[:end],
                              weights=self.batch_sizes[:end])

    def is_converged(self, min_samples=32, converge_deviation=.01):
        if len(self.items) < min_samples:
            return False

        stddev = np.std(self._means[-min_samples:])
        avg = np.mean(self._means[-min_samples:])
        is_cvg = stddev / avg < converge_deviation

        if not is_cvg:
            print(
                f'Convergence RMSD {stddev} / AVG {avg} = {stddev / avg}'
                f' >= {converge_deviation}')
        else:
            print(
                f'Converged! RMSD {stddev} / AVG {avg} = {stddev / avg}'
                f' < {converge_deviation}')

        return is_cvg

    def __str__(self):
        return self.accuracy.__str__()


def save(path,
         model,
         optimizer,
         batch,
         rng_states=None,
         verbose=True,
         allow_overwrite=False,
         **kwargs):
    save_dict = {'model_state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'batch': batch,
                 'extra': kwargs}

    if rng_states is None:
        rng_states = {
            'torch_seed': torch.initial_seed(),
            'torch_cuda_seed': torch.cuda.initial_seed(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state(),
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
        }
    save_dict['rng_states'] = rng_states

    fname = path.format(str(batch).zfill(8))
    if verbose:
        print(f"Saving to {fname}...")

    if not allow_overwrite:
        from pathlib import Path
        if Path(fname).exists():
            raise FileExistsError("I'm not allowing you to overwrite previous"
                                  " checkpoints.")

    torch.save(save_dict, fname)


def _load_checkpoint(fname='ckpt.pth', device='cpu', verbose=False):
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Checkpoint at `{fname}` not found.")

    if verbose:
        print(f"Loading from {fname} to {device}.")

    return torch.load(fname, map_location=device)


def restore(model,
            fname: str,
            device: str = 'cuda',
            optimizer_class=None,
            deterministic: bool = False,
            verbose: bool = False):
    ckpt = _load_checkpoint(fname, verbose=verbose)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    output = [model]
    if optimizer_class is not None:
        optimizer = optimizer_class(model.parameters())
        optimizer.load_state_dict(ckpt['optimizer'])
        output.append(optimizer)
    if deterministic:
        if 'rng_states' not in ckpt:
            warnings.warn("Checkpoint does not contain RNG states, maybe"
                          " saved in an old version of this application.")
        else:
            rng_states = ckpt['rng_states']
            torch.manual_seed(rng_states['torch_seed'])
            torch.cuda.manual_seed(rng_states['torch_cuda_seed'])
            torch.cuda.manual_seed_all(rng_states['torch_cuda_seed'])
            torch.set_rng_state(rng_states['torch_rng_state'])
            torch.cuda.set_rng_state(rng_states['torch_cuda_state'])
            # seeds are not retrievable by random and np.random, but that means
            # we also don't use them (while we would use torch.get_initial_seed())
            random.setstate(rng_states['random_state'])
            np.random.set_state(rng_states['np_random_state'])

    return tuple(output), ckpt


def eval_dataloader(
    model,
    eval_pairs,
    eval_batches: int = None,
    eval_convergence_rmsd: float = None,
    accuracy=RMSE,
    on_cpu=False,
    eval_cvg_len: int = 16):
    # a dictionary for the noisy and denoised accuracies
    noisy_aggregate = AccuracyAggregate(accuracy())
    denoised_aggregate = AccuracyAggregate(accuracy())

    model = model.cpu() if on_cpu else model.cuda()
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(eval_batches),
                      total=eval_batches,
                      desc="Evaluating..."):
            movie, target, _ = next(eval_pairs)
            movie_eval = movie.cuda(
                non_blocking=True) if not on_cpu else movie.cpu()
            target_eval = target.cuda(
                non_blocking=True) if not on_cpu else target.cpu()

            baseline = model._extract_attack_frame(movie_eval)
            out = model(movie_eval)

            # we add a denoised batch and noisy batch to the accuracy agg
            model.accuracy(baseline, target_eval, noisy_aggregate)
            model.accuracy(out, target_eval, denoised_aggregate)

            i += 1
            if eval_convergence_rmsd is not None:
                # we need the accuracy to converge, we consider it
                # converged if the last `eval_cfg_len` batches
                # did not change the computed accuracy too much,
                # e.g. the mean deviation of the average from the acc
                # in the last 10 batches should be within
                # a given tolerance of .05.
                is_converged = denoised_aggregate.is_converged(
                    eval_cvg_len,
                    eval_convergence_rmsd)
                if not is_converged:
                    break  # we need only one False to continue

        return noisy_aggregate, denoised_aggregate


def find_checkpoints(path, pattern="ckpt_batch(\d+)\.ckpt", max_amount=None):
    import re
    regex = re.compile(pattern)
    fnames_full = sorted([str(p) for p in Path(path).iterdir()])
    fnames = sorted([f.name for f in Path(path).iterdir()])
    fnames_matched = []
    fnames_full_matched = []
    batches_matched = []
    for i, (f, ff) in enumerate(zip(fnames, fnames_full)):
        match = regex.match(f)
        if not match:
            continue
        batch = int(match.group(1))
        fnames_matched.append(f)
        fnames_full_matched.append(ff)
        batches_matched.append(batch)

    if max_amount is not None:
        max_amount = np.min((max_amount, len(fnames_matched)))
        fnames_matched = fnames_matched[::len(fnames_matched) // max_amount]
        fnames_full_matched = fnames_full_matched[::len(fnames_matched) // max_amount]
        batches_matched = batches_matched[::len(batches_matched) // max_amount]

    return {b: f for f, b in zip(fnames_full_matched, batches_matched)}