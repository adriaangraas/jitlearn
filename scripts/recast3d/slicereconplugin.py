import slicerecon  # important! must be first line to avoid GLIBC error
import bisect
import collections
import os
from time import sleep
import timeit

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({'figure.raise_window': False})
plt.style.use('dark_background')

import numpy as np
import torch
from abc import ABC, abstractmethod

from mpl_toolkits.axes_grid1 import ImageGrid

from spatiotemporal.rayveds import BufferingMultiProcessingDataLoaderIter


class SliceReconPlugin(ABC):
    """Plug-in for RECAST3D."""

    def __init__(self,
                 expt,
                 verbose=True):
        self.verbose = verbose

        self.t = -1  # discrete timestep update (internal)
        self.time = None
        self.shape = None

        # Keys are the `self.t` that the images/flows are computed for, not
        # the timesteps that they are computed at.
        self.img_fbp = {}  # ims incoming
        self.img_denoised = {}
        self.expt = expt
        self.time_start = None

        self.axs = []

    @abstractmethod
    def _get_current_model(self):
        pass

    def __call__(self, shape, xs, slice_idx, *args, **kwargs):
        """
        This method directly receives the incoming call from SliceRecon
        and calls _slice_estimate to estimate the slices.

        :param args:
        :param kwargs:
        :return:
        """
        self.t += 1

        if self.verbose:
            print(f"\n* SliceReconPlugin t={self.t}:")

        time_in = timeit.default_timer()  # timer starts

        # This is to prevent SliceRecon initialization packets from being processed here
        if shape[1] == 1 or shape[0] == 1:
            print("Call ignored because shape[0] or shape[1] == 1.")
            return [shape, xs]

        # Make sure that the shapes of reconstructions cannot change during a run
        if self.shape is None:
            self.shape = shape
        elif self.shape != shape:
            raise NotImplementedError("Shape may not change during a run.")

        if slice_idx == 0 or slice_idx == 1 or slice_idx % 2 != 0:
            # raise NotImplementedError(
            #     "Slice index may not change during a run.")
            # return [shape, xs]
            return self._pass_through(xs, shape)

        SLICERECON_GROUP_SIZE = 50
        if self.time is None:
            # starting experiment time, first reconstruction is only made
            # after a first reconstruction
            self.time = self.expt.SLEEP_TIME \
                        + self.expt.time_per_proj * SLICERECON_GROUP_SIZE
            self.time_start = timeit.default_timer()
        else:
            time_diff = timeit.default_timer() - self.time_start
            self.time = self.expt.SLEEP_TIME \
                        + self.expt.time_per_proj * SLICERECON_GROUP_SIZE \
                        + time_diff

        print(f"Current experiment time: {self.time - self.expt.SLEEP_TIME}")

        # Add image to history
        self.img_fbp[self.t] = self._parse_in(xs, shape)
        # self.img_fbp[self.t].setflags(write=False)

        model = self._get_current_model()
        if model is False:
            print("No model stored for current time. Ignoring slice.")
            return self._pass_through(xs, shape)

        print("Running model")
        movie = self.img_fbp[self.t]
        self.img_denoised[self.t] = self._call_model(model, movie)

        sl1 = slice(90, 250)
        sl2 = slice(100, 260)
        if len(self.axs) == 0:
            fbp = movie[sl1, sl2]
            denoised = self.img_denoised[self.t][sl1, sl2]
            fig1 = plt.figure(1)
            fig1.set_facecolor('#191919')  # matching RECAST3D
            self.axs.append(plt.imshow(fbp, interpolation=None,
                                       vmin=0.0, vmax=0.75))
            plt.draw()

            fig3 = plt.figure(3)
            fig3.set_facecolor('#191919')  # matching RECAST3D
            self.axs.append(plt.imshow(denoised, interpolation=None,
                                       vmin=0.0, vmax=0.75))
            plt.pause(.0001)
        elif self.t % 2 == 0:
            plt.figure(1)
            fbp = movie[sl1, sl2]
            self.axs[0].set_data(fbp)
            plt.draw()

            plt.figure(3)
            denoised = self.img_denoised[self.t][sl1, sl2]
            self.axs[1].set_data(denoised)
            plt.pause(.0001)

        out = self._get_out(self.img_denoised[self.t])
        time_out = timeit.default_timer()
        if self.verbose:
            print(f"Runtime: {time_out - time_in}")

        return out

    @classmethod
    def _pass_through(cls, xs, shp):
        return cls._get_out(cls._parse_in(xs, shp))

    @classmethod
    def _parse_in(cls, xs, shp):
        return np.array(xs).reshape(shp).astype(np.float32) / 7500

    @classmethod
    def _get_out(cls, im):
        np.clip(im, a_min=0., a_max=None, out=im)
        return [im.shape, im.ravel().tolist()]

    def _call_model(self, model, movie):
        movie = torch.from_numpy(movie)
        movie = torch.unsqueeze(movie, 0)  # add batch
        movie = torch.unsqueeze(movie, 0)  # add time
        movie = torch.unsqueeze(movie, -1)  # add z
        movie_eval = movie.cuda(non_blocking=True)
        out = model(movie_eval, {})
        return torch.squeeze(out).detach().cpu().numpy()

    @abstractmethod
    def add_slice(self, slice_id, orientation):
        print(f"TODO:Not adding slice {slice_id}")
        print(slice_id)
        print(orientation)

    def remove_slice(self, slice_id):
        print(f"Removing slice {slice_id}")


class PretrainedSliceReconPlugin(SliceReconPlugin):
    def __init__(self):
        # Find all checkpoints
        match_fnames, match_times = expt.find_checkpoints(basedir)
        print(match_fnames)

        self.checkpoints = collections.OrderedDict()
        for batch, fname in match_fnames.items():
            # load model
            time = match_times[batch]
            curr_num = int(time // expt.time_per_proj) + expt.slice_num_start
            self.checkpoints[time] = {
                'fname': fname,
                'curr_num': curr_num,
            }

    def _get_current_model(self):
        assert self.time is not None

        times = list(self.checkpoints.keys())
        if self.time < times[0]:
            return False  # no model found

        ckpt_indx = bisect.bisect_left(times, self.time)
        ckpt_time = times[ckpt_indx]
        ckpt_fname = self.checkpoints[ckpt_time]['fname']
        try:  # ignore uninitialized
            if self.expt.fname == ckpt_fname:
                return self.expt.model
        except AttributeError:
            pass

        kwargs = {'parent_dir': basedir,
                  'fname': ckpt_fname}
        print(f"Loading {ckpt_fname}...")
        self.expt = self.expt.resume(**kwargs)
        self.expt.fname = ckpt_fname  # store so that we can check later
        self.expt.model.to('cuda')
        self.expt.model.eval()
        return self.expt.model


class RealtimeSliceReconPlugin(SliceReconPlugin):
    def __init__(self, expt, state, orientation):
        super().__init__(expt)
        torch.cuda.set_device(1)
        self.model = expt.model
        self.state = state
        self.orientation = orientation

    def _get_current_model(self):
        self.model.load_state_dict(self.state)
        self.model.cuda()
        return self.model

    def add_slice(self, slice_id, orientation):
        print(f"Adding slice {slice_id}")
        if slice_id % 2 == 0:
            self.orientation[...] = torch.asarray(orientation)


def realtime(model, params, opt, pairs, buffer_size):
    # They *must* be started at the same time!
    pairs_iter = BufferingMultiProcessingDataLoaderIter(pairs, buffer_size)
    pairs_train = enumerate(pairs_iter)

    for i, train_item in pairs_train:
        movie, target, signal = train_item

        model.train()
        opt.zero_grad()
        out = model(movie.cuda(), {k: v.cuda() for k, v in signal.items()})
        loss = model.crit(out, target.cuda())
        loss.backward()
        opt.step()
        model.cpu()
        params.update(model.state_dict())
        model.cuda()

        # im[...] = torch.vstack([torch.hstack([torch.asarray(im[0, ...])
        #     for im in movie[4 * j:4 * j + 4]])
        #     for j in range(4)])[..., 0]

    fig = plt.figure(2)
    fig.set_facecolor('#191919')  # matching RECAST3D
    # grid = ImageGrid(fig2, 111, nrows_ncols=(4, 4), axes_pad=0.1)
    objs = []
    objs = None
    def update(i, objs=objs):
        # for i, train_item in pairs_train:
        movie = None
        for j in range(10):
            train_item = next(pairs_train)
            _, (movie, target, signal) = train_item

            model.train()
            opt.zero_grad()
            out = model(movie.cuda(), {k: v.cuda() for k, v in signal.items()})
            loss = model.crit(out, target.cuda())
            loss.backward()
            opt.step()
            model.cpu()
            params.update(model.state_dict())
            model.cuda()

        # im = np.vstack([np.hstack([im[0, ...] for im in movie[4*j:4*j+4]])
        #            for j in range(4)])

        if objs is None:
            # objs = plt.gca().imshow(im, interpolation=None, vmin=0.0, vmax=0.75)
            objs = plt.gca().imshow(movie[0, 0], interpolation=None, vmin=0.0, vmax=0.75)
        else:
            # objs.set_data(im)
            objs.set_data(movie[0])

        # assert len(movie) == 32
        # if len(objs) == 0:
        #     for ax, im in zip(grid, movie[:16]):
        #         objs.append(ax.imshow(im[0, ...], interpolation=None,
        #                               vmin=0.0, vmax=0.75))
        #     for ax, obj in zip(grid, objs):
        #         ax.draw_artist(obj)
        # else:
        #     for ax, obj, im in zip(grid, objs, movie[:16]):
        #         obj.set_data(im[0, ...])
        #         ax.draw_artist(obj)

        return [objs]

    anim = FuncAnimation(fig, update, None, interval=5, blit=True)
    plt.show()


def start_training(expt, model, params):
    torch.cuda.set_device(0)
    model.cuda()
    realtime(
        model,
        params,
        expt.opt,
        expt.pairs_train,
        expt.buffer_size)


if __name__ == '__main__':
    # Note: include the examples directory in the PYTHONPATH
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)

    from dae import RealtimeTopToBottomTest

    basedir = "/export/scratch2/adriaan/zond_fastdvdnet_training"

    orientation = torch.zeros((9,))
    orientation.share_memory_()
    expt = RealtimeTopToBottomTest(orientation)
    model = expt.model
    state = torch.multiprocessing.Manager().dict(model.state_dict())

    # train in a separate process
    p = mp.Process(target=start_training, args=(expt, model, state))
    p.start()

    plugin = RealtimeSliceReconPlugin(expt, state, orientation)
    x = np.ones((100, 100))
    # shp, xs = [x.shape, x.ravel().tolist()]
    # for i in range(100):
    #     shp, ys = plugin(shp, xs, 0)
    #     sleep(10.)

    # y = np.array(ys).reshape(shp)
    p = slicerecon.plugin("tcp://*:5652", "tcp://localhost:5555")
    p.slice_data_callback(plugin)
    p.set_slice_callback(plugin.add_slice)
    p.remove_slice_callback(plugin.remove_slice)
    p.listen()
