import os
import itertools
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
        from torch.nn.functional import mse_loss
        return mse_loss(output, target).cpu().numpy().item()
        # return (
        #     torch.sqrt(
        #         torch.square(
        #             torch.subtract(output, target)
        #         ).mean()
        #     ).cpu().numpy()
        # )

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
         time=None,
         rng_states=None,
         verbose=True,
         allow_overwrite=False,
         **kwargs):
    save_dict = {'model_state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'batch': batch,
                 'time': time}
    save_dict.update(kwargs)

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

        # for noisy, denoised in zip(noisy_aggregates, denoised_aggregates):
        #     print(f"{noisy} eval noisy: {noisy.average()}")
        #     print(f"{denoised} eval denoised: {denoised.average()}")
        #     print(f"Ratio: {denoised.average() / noisy.average()}")
        #     if summary_writer is not None:
        #         summary_writer.add_scalar(
        #             f'/{noisy}/eval_noisy',
        #             noisy.average(),
        #             )
        #         summary_writer.add_scalar(
        #             f'/{denoised}/eval_denoised',
        #             denoised.average(),
        #             cumulative_batch)
        #         summary_writer.add_scalar(
        #             f'/{denoised}/denoised-to-noisy',
        #             denoised.average() / noisy.average(),
        #             cumulative_batch)

        return noisy_aggregate, denoised_aggregate


def find_checkpoints(path, pattern="ckpt_batch(\d+)\.ckpt", max_amount=None):
    import re
    match_fnames = {}
    regex = re.compile(pattern)
    fnames = sorted([str(p) for p in Path(path).iterdir()])
    if max_amount is not None:
        max_amount = np.min((max_amount, len(fnames)))
        fnames = fnames[::len(fnames) // max_amount]  # subsample for speed
    for i, file in enumerate(fnames):
        match = regex.match(file)
        if not match:
            continue
        batch = int(match.group(1))
        match_fnames[batch] = file
    return match_fnames

# class Trainer:
#     def __init__(self):
#         # state variables
#         self.epoch = -1
#         self.cumulative_batch = -1
#
#     def realtime(self,
#                  model: SpatioTemporalModel,
#                  opt: torch.optim.Adam,
#                  pairs,
#                  plot: bool = False,
#                  pairs_eval=(),
#                  info_every: int = 100,
#                  eval_max_len: int = 1,
#                  eval_convergence_rmsd: float = None,
#                  eval_accuracies: list = (),
#                  eval_on_cpu: bool = False,
#                  summary_writer: SummaryWriter = None,
#                  verbosity_level: int = 1,
#                  plot_fn=_default_plot_fn,
#                  save_fn=None,
#                  clip_grad_norm=None,
#                  pairs_visu=None,
#                  buffer_size=1000):
#         """Run `epochs` iterations of `pairs` on `model`."""
#         self.verbosity_level = verbosity_level
#
#         # They *must* be started at the same time!
#         pairs_iter = BufferingMultiProcessingDataLoaderIter(pairs, buffer_size)
#         pairs_train = enumerate(pairs_iter, self.cumulative_batch + 1)
#         try:
#             pairs_eval = BufferingMultiProcessingDataLoaderIter(pairs_eval)
#         except:
#             pairs_eval = iter(pairs_eval)
#
#         def _curtime():
#             import time
#             return time.time() - pairs_iter._dataset.start_time
#
#         trains_times = []
#         trains = []
#         evals_times = []
#         evals = []
#
#         for train_item, eval_item in itertools.zip_longest(
#             pairs_train, pairs_eval):
#
#             if train_item is not None:
#                 trains_times.append(_curtime())
#                 self.cumulative_batch, (movie, target, signal) = train_item
#                 signal_cuda = {k: v.cuda() for k, v in signal.items()}
#
#                 model.train()
#                 opt.zero_grad()
#                 out = model(movie.cuda(), signal_cuda)
#                 loss = model.crit(out, target.cuda())
#                 trains.append(model.accuracy(out, target.cuda(), RMSE()))
#                 loss.backward()
#                 if clip_grad_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), clip_grad_norm)
#                 opt.step()
#
#                 if save_fn is not None and self.cumulative_batch > 0:
#                     save_fn(self.cumulative_batch, _curtime())
#
#                 if verbosity_level > 0:
#                     self._print_status(f"\nLoss: {loss}")
#
#                 if summary_writer is not None:
#                     summary_writer.add_scalar(
#                         '/Loss/train', loss, self.cumulative_batch)
#                     summary_writer.add_scalar(
#                         '/Extra/lr', opt.defaults['lr'], self.cumulative_batch)
#                     summary_writer.add_scalar(
#                         '/Extra/epoch', self.epoch, self.cumulative_batch)
#                     summary_writer.add_scalar(
#                         '/Extra/batch', self.cumulative_batch,
#                         self.cumulative_batch)
#
#                 if verbosity_level > 1:
#                     print('.', end='')
#
#                 if verbosity_level > 0:
#                     print('\n', end='')
#
#                 if plot:
#                     plot_fn('train-out', out.cpu().detach().numpy())
#                     plt.pause(1.)
#
#             if eval_item is not None:
#                 eval_mov, eval_tar, _ = eval_item
#                 evals_times.append(_curtime())
#                 noisy_aggs, denoised_aggs = eval_dataloader(
#                     model,
#                     pairs_eval,
#                     # eval_max_len if train_item is None else 1,
#                     eval_max_len,
#                     eval_convergence_rmsd,
#                     eval_accuracies,
#                     eval_on_cpu,
#                     plot,
#                     summary_writer,
#                     plot_fn=plot_fn)
#                 # assert len(denoised_aggs[0].items) == 1
#                 evals.append(denoised_aggs[0].average())
#
#                 # model.cuda()
#                 # model.eval()
#                 # with torch.no_grad():
#                 #     out_ev = model(eval_mov.cuda())
#                 #     evals.append(model.accuracy(out_ev, eval_tar.cuda(), RMSE()))
#
#             if plot:
#                 # if (len(evals) + len(trains)) % 5 == 0:
#                 #     plt.figure('losses')
#                 #     plt.cla()
#                 #     plt.ylim(0, 0.005)
#                 #     plt.plot(trains_times, trains, label='loss')
#                 #     plt.plot(evals_times, evals, label='eval')
#                 #     plt.legend()
#                 #     plt.pause(.0001)
#
#                 if pairs_visu is not None:
#                     model.eval()
#                     for (movie, target, signal) in pairs_visu:
#                         movie = torch.tensor(movie)
#                         movie_visu = movie.cuda(
#                             non_blocking=True) if not eval_on_cpu else movie.cpu()
#                         signal_visu = {
#                             k: v.cuda() if not eval_on_cpu else v.cpu()
#                             for k, v in signal.items()}
#
#                         # note: is also evaluated on CPU, if eval mode is
#                         out_visu = model(movie_visu, signal_visu)
#                         plot_fn('visu-in', movie_visu.cpu().detach().numpy())
#                         plot_fn('visu-out', out_visu.cpu().detach().numpy())
#                         # _target_visu = target[np.newaxis, ..., 0]
#                         # _baseline_visu = model._extract_attack_frame(
#                         #     movie).cpu().detach().numpy()
#                         # plot_fn('visu-(out-target)',
#                         #         out_visu.cpu().detach().numpy()
#                         #         - _target_visu)
#                         # plot_fn('visu-(in-target)',
#                         #         _baseline_visu - _target_visu)
#                         # plot_fn('visu-denoised-to-noise',
#                         #         np.true_divide(
#                         #             out_visu.cpu().detach().numpy() - _target_visu,
#                         #             _baseline_visu - _target_visu
#                         #         ))
#
#             if eval_on_cpu:
#                 model.cuda()  # after eval move model back to GPU
#
#         plt.figure()
#         plt.pause(100000)
#
#     def train(self,
#               model: SpatioTemporalModel,
#               opt: torch.optim.Adam,
#               pairs,
#               epochs: int,
#               plot: bool = False,
#               pairs_eval=(),
#               info_every: int = 100,
#               eval_max_len: int = None,
#               eval_convergence_rmsd: float = None,
#               eval_accuracies: list = (),
#               eval_on_cpu: bool = False,
#               summary_writer: SummaryWriter = None,
#               verbosity_level: int = 1,
#               plot_fn=_default_plot_fn,
#               save_fn=None,
#               clip_grad_norm=None,
#               with_scheduler=False,
#               pairs_visu=None,
#               max_batches=None):
#         """Run `epochs` iterations of `pairs` on `model`.
#
#         :param model:
#         :param opt:
#         :param pairs:
#         :param epochs:
#         :param plot:
#         :param pairs_eval:
#         :param info_every:
#         :param eval_max_len:
#         :param summary_writer:
#         :return:
#         """
#         self.verbosity_level = verbosity_level
#
#         # if isinstance(pairs_eval, DataLoader):
#         #     # if __iter__ is called in every `_eval_dataloader`, high startup
#         #     # costs may be incurred, such as is the case for our live
#         #     # reconstruction dataloader (that cannot do hard labour in
#         #     # __init__ because of multithreading starting only after __iter__).
#         #     pairs_eval = iter(pairs_eval)
#
#         pairs_train_gen = lambda: enumerate(pairs, self.cumulative_batch + 1)
#         pairs_eval_gen = lambda: iter(pairs_eval)
#
#         # do not restart datasets if we have enough GPUs
#         not_overlapping_gpus = False
#         if pairs_eval != ():
#             try:
#                 train_devs = pairs.uniform.device_ids
#                 eval_devs = pairs_eval.dataset.device_ids
#                 not_overlapping_gpus = not any(
#                     id in eval_devs for id in train_devs)
#             except AttributeError:
#                 pass
#         if pairs_eval == () or not_overlapping_gpus:
#             _pairs_train_enumerator = pairs_train_gen()
#             _pairs_eval_iter = pairs_eval_gen()
#             pairs_train_gen = lambda: _pairs_train_enumerator
#             pairs_eval_gen = lambda: _pairs_eval_iter
#
#         epochs_iter = itertools.count(self.epoch + 1)
#         if epochs is not None:
#             epochs_iter = range(self.epoch + 1, self.epoch + 1 + epochs)
#
#         if with_scheduler:
#             scheduler = ReduceLROnPlateau(opt)
#
#         starter = torch.cuda.Event(enable_timing=True)
#         ender = torch.cuda.Event(enable_timing=True)
#
#         def _do_eval_and_visu():
#             # end of epoch, do evaluation
#             if pairs_eval != ():
#                 noisy, denoised = _eval_dataloader(model,
#                                                         pairs_eval_gen(),
#                                                         eval_max_len,
#                                                         eval_convergence_rmsd,
#                                                         eval_accuracies,
#                                                         eval_on_cpu,
#                                                         plot,
#                                                         summary_writer,
#                                                         plot_fn=plot_fn)
#                 if with_scheduler:
#                     # TODO: the first metric is used for determining
#                     #     the LR, by convention. Not really clear.
#                     scheduler.step(noisy[0].average())
#
#             if pairs_visu is not None:
#                 model.eval()
#                 for (movie, target, signal) in pairs_visu:
#                     movie_visu = movie.cuda(
#                         non_blocking=True) if not eval_on_cpu else movie.cpu()
#                     signal_visu = {
#                         k: v.cuda() if not eval_on_cpu else v.cpu()
#                         for k, v in signal.items()}
#
#                     # note: is also evaluated on CPU, if eval mode is
#                     out_visu = model(movie_visu, signal_visu)
#
#                     mv = movie_visu.cpu().detach()
#                     mvfig = plot_fn('visu-in',
#                                     model._extract_attack_frame(mv).numpy())
#                     ov = out_visu.cpu().detach().numpy()
#                     tv = target.cpu().detach().numpy()
#                     ovfig = plot_fn('visu-out', ov)
#                     tvfig = plot_fn('visu-target', tv)
#                     if summary_writer is not None:
#                         summary_writer.add_figure('visu-in', mvfig,
#                                                   self.cumulative_batch)
#                         # summary_writer.add_images('visu-in-im',
#                         #                           self._to_image(mv),
#                         #                           self.cumulative_batch)
#                         summary_writer.add_figure('visu-out', ovfig,
#                                                   self.cumulative_batch)
#                         summary_writer.add_figure('visu-target', tvfig,
#                                                   self.cumulative_batch)
#                     if plot is not None:
#                         plt.pause(.5)
#
#         for self.epoch in epochs_iter:
#             _do_eval_and_visu()
#             for self.cumulative_batch, (
#             movie, target, signal) in pairs_train_gen():
#                 starter.record()
#                 model.train()
#                 opt.zero_grad()
#                 signal_cuda = {k: v.cuda() for k, v in signal.items()}
#                 out = model(movie.cuda(), signal_cuda)
#                 loss = model.crit(out, target.cuda())
#                 loss.backward()
#                 if clip_grad_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(
#                         model.parameters(), clip_grad_norm)
#                 opt.step()
#                 ender.record()
#
#                 if save_fn is not None and self.cumulative_batch > 0:
#                     save_fn(self.cumulative_batch)
#
#                 if verbosity_level > 0:
#                     print(f"\nLoss: {loss}")
#
#                 if summary_writer is not None:
#                     summary_writer.add_scalar(
#                         'Extra/iteration_time',
#                         starter.elapsed_time(ender),
#                         self.cumulative_batch
#                     )
#                     summary_writer.add_scalar(
#                         '/Loss/train', loss, self.cumulative_batch)
#                     if with_scheduler:
#                         summary_writer.add_scalar(
#                             '/Extra/lr', scheduler.get_last_lr(),
#                             self.cumulative_batch)
#                     else:
#                         summary_writer.add_scalar(
#                             '/Extra/lr', opt.defaults['lr'],
#                             self.cumulative_batch)
#                     summary_writer.add_scalar(
#                         '/Extra/epoch', self.epoch, self.cumulative_batch)
#                     summary_writer.add_scalar(
#                         '/Extra/batch', self.cumulative_batch,
#                         self.cumulative_batch)
#                     model.write_summary(summary_writer,
#                                         self.cumulative_batch)
#
#                 if verbosity_level > 1:
#                     print('.', end='')
#
#                 if verbosity_level > 0:
#                     print('\n', end='')
#
#                 # if plot:
#                 #     plot_fn('train-out', out.cpu().detach().numpy())
#
#                 if self.cumulative_batch % info_every == 0:
#                     # Training accuray is removed, because I didn't want to
#                     # implement a "RandomCrop + ROI" implementation for now.
#                     # if getattr(model, 'accuracy', False):
#                     #     psnr = model.accuracy(out, target)
#                     #     print(f"PSNR: {model.accuracy(out, target)}")
#                     #     if summary_writer is not None:
#                     #         summary_writer.add_scalar(
#                     #             '/Accuracy/train', psnr, self.cumulative_batch)
#                     #
#                     if plot:
#                         plot_fn('in', movie.detach().numpy())
#                         plot_fn('target', target.detach().numpy())
#                         plot_fn('out', out.cpu().detach().numpy())
#
#                     # break at end of loop to not spill a draw of the dataloader
#                     if self.cumulative_batch > 0:
#                         break
#             # endfor
#             if eval_on_cpu:
#                 model.cuda()  # after eval move model back to GPU
#
#             if max_batches is not None and self.cumulative_batch >= max_batches:
#                 _do_eval_and_visu()
#                 return
#
#     def _to_image(self, tensor):
#         if tensor.ndim == 5:
#             im = tensor[0].permute(3, 0, 1, 2)
#         else:
#             im = tensor[0]
#         return im
#
#     def eval(self,
#              model: SpatioTemporalModel,
#              pairs_iter=(),
#              plot: bool = False,
#              max_len: int = None,
#              convergence_rmsd=None,
#              accuracies: list = (),
#              on_cpu: bool = False,
#              summary_writer: SummaryWriter = None,
#              verbosity_level: int = 1,
#              plot_fn: Callable = _default_plot_fn):
#         self.verbosity_level = verbosity_level
#         eval_dataloader(model,
#                         pairs_iter,
#                         max_len, convergence_rmsd,
#                         accuracies, on_cpu, plot, summary_writer)
#         if verbosity_level > 0:
#             print('\n', end='')
#
#     def timings(self,
#                 model: SpatioTemporalModel,
#                 pairs=(),
#                 on_cpu: bool = False,
#                 max_len: int = 25,
#                 warm_up_len: int = 10,
#                 train=False):
#         starter = torch.cuda.Event(enable_timing=True)
#         ender = torch.cuda.Event(enable_timing=True)
#         timings = np.zeros((max_len, 1))
#
#         model = model.cpu() if on_cpu else model.cuda()
#         if train:
#             model.train()
#         else:
#             model.eval()
#
#         movie, target, signal = next(iter(pairs))
#         # with torch.no_grad():  # re-enable this in case of memory issues
#
#         # warm-up
#         print("Warming up...")
#         for i in range(warm_up_len):
#             if model.attack_frame == model.sequence_length:
#                 movie = movie[:, :-1, ...]  # this is the input
#
#             movie = movie.cuda() if not on_cpu else movie.cpu()
#             signal_cuda = {k: v.cuda() for k, v in signal.items()}
#             _ = model(movie, signal_cuda)
#
#         for i in tqdm(range(max_len)):
#             if model.attack_frame == model.sequence_length:
#                 movie = movie[:, :-1, ...]  # this is the input
#
#             movie = movie.cuda() if not on_cpu else movie.cpu()
#             signal_cuda = {k: v.cuda() for k, v in signal.items()}
#             starter.record()
#             out = model(movie, signal_cuda)
#             if train:
#                 loss = model.crit(out, target.cuda())
#                 loss.backward()
#             ender.record()
#
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)
#             timings[i] = curr_time
#             # print(f"{i}/{max_len}: {timings[i]}")
#
#         print(f"Shape: {movie.shape}")
#         mean_syn = np.sum(timings) / max_len
#         std_syn = np.std(timings)
#         print(f"Mean: {mean_syn}")
#         print(f"Std dev.: {std_syn}")
#         return mean_syn, std_syn
