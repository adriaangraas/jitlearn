import itertools
import random
import threading
import time
import warnings
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Sequence

import astrapy.kernels
import cupy as cp
import cupy.cuda
import numpy as np
import reflex
import torch
from astrapy import Detector, ProjectionGeometry, VolumeGeometry
from reflex import centralize
from torch.utils.data import IterableDataset

from jitlearn.buffer import ProjectionBufferBp


class SampleIntervalTooSmallError(Exception):
    pass


class TabletDataset(IterableDataset):
    """
    Dissolving tablet in fluid dataset, acquired at the FleX-ray laboratory
    at CWI, Amsterdam. Generating samples from this dataset requires an
    installation of my _Reflex_ package, to parse data from the FleX-ray
    lab, and of _AstraPy_, for generation of reconstructions from a buffer.

    This subclass uses a static buffer to sample from. Have a look at
    ContinuousTabletDataset and RealtimeTabletDataset for updating buffers.
    """

    def __init__(self,
                 projs_path: str,
                 settings_path: str,
                 proj_ids: range,
                 voxels: Any,
                 pos_min,
                 pos_max,
                 rot_min,
                 rot_max,
                 darks_path: str = None,
                 flats_path: str = None,
                 iso_voxel_size: float = None,
                 device_ids=None,
                 reconstruction_rotation_intval: float = np.pi * 2,
                 nr_angles_per_rot: int = 150,
                 reco_interval: int = 150,
                 sequence_length: int = 1,
                 sequence_stride: int = 20,
                 corrections: Any = True,
                 verbose: bool = False,
                 squeeze: bool = False,
                 preload: bool = False,
                 start_time: float = None,
                 n2i_mode: str = 'even_odd',
                 texture_type='array2d'):
        """
        Initialize the `TabletDataset`.

        Note that everything in __init__ needs to be pickle-able! So complicated
        or lambda objects go in __iter__.

        :param projs_path: path passed to Reflex
        :param settings_path: path passed to Reflex
        :param voxels: number of voxels for reconstruction (3-tuple or scalar)
        :param proj_ids: range of projection numbers that the buffer may load
        :param darks_path: path passed to Reflex
        :param flats_path: path passed to Reflex
        :param iso_voxel_size: voxel size (in mm), scalar, be isotropic
        :param device_ids: this needs to be passed so that the multithreaded
            worker function knows on which GPU to launch the dataset.
        :param reconstruction_rotation_intval:
        :param nr_angles_per_rot: number of angles used in acquisition
        :param reco_interval: how many projections to use for reconstruction
        :param sequence_length: how many frames to return per sample
        :param sequence_stride: the distance in projections between consecutive
            frames
        :param corrections: corrections to pass to Reflex
        :param verbose:
        :param preload: start the dataset by loading everything into RAM
            and time.
        :param n2i_mode: 'even_odd', 'even_odd_mixed'
        """

        self.settings = reflex.Settings.from_path(settings_path)
        self.projs_path = projs_path
        if darks_path is None:
            darks_path = settings_path
        self.darks_path = darks_path
        if flats_path is None:
            flats_path = settings_path
        self.flats_path = flats_path
        self.corrections = corrections
        self.device_ids = device_ids
        self.seq_len = sequence_length
        self.squeeze = squeeze
        self.preload = preload
        self.n2i_mode = n2i_mode
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.texture_type = texture_type

        assert sequence_stride % 2 == 0, (
            "If stride is not even, information from even reconstructions "
            "could spill into odd reconstructions.")
        self.seq_stride = sequence_stride

        if np.isscalar(voxels):
            voxels = [voxels] * 2 + [1]  # a slice with 1 voxel thickness
        self.voxels = voxels

        if proj_ids is None:
            proj_ids = np.array(list(reflex.proj._matches(
                projs_path, reflex.proj.PROJECTION_FILE_REGEX).keys()))
        self.proj_ids = proj_ids

        angles = (np.array(self.proj_ids)
                       * reconstruction_rotation_intval / nr_angles_per_rot)
        if reco_interval is None:
            reco_interval = int(
                np.ceil(
                    2 * np.pi * nr_angles_per_rot / reconstruction_rotation_intval))
        self.reco_interval = reco_interval

        geometry = self.reflex_geom(self.settings, angles,
                                    self.corrections)
        self.geometries = dict()
        for id, g in zip(self.proj_ids, geometry):
            self.geometries[id] = g

        self.voxel_size = iso_voxel_size
        self.vol_ext_min = -iso_voxel_size * np.array(voxels) / 2
        self.vol_ext_max = iso_voxel_size * np.array(voxels) / 2
        self.verbose = verbose
        self.start_time = start_time

    def __iter__(self):
        """
        This is called when the dataloader takes the dataset, and is executed
        in a multiprocessing thread. This is executed on a dedicated GPU if
        the dataset is initialized with the _worker_init_fn in this module.
        """
        if self.verbose:
            print("Setting up for darkfielding and flatfielding...")

        # preloading all the necessary darks and whites from disk
        darks = reflex.darks(self.darks_path, verbose=self.verbose)
        whites = reflex.flats(self.flats_path, verbose=self.verbose)
        if len(darks) != 0:
            if len(darks) > 1:
                darks = darks.mean(0)
            dark = np.squeeze(np.array(darks))
        if len(whites) != 0:
            if len(whites) > 1:
                whites = whites - darks if len(darks) != 0 else whites
                whites = whites.mean(0)
            white = np.squeeze(np.array(whites))

        def _preprocess_projs(projs: cp.ndarray):
            # remove darkfield from projections
            cp.subtract(projs, cp.array(dark), out=projs)
            cp.divide(projs, cp.array(white),
                      out=projs)  # flatfield the projections
            cp.log(projs, out=projs)  # take -log to linearize detector values
            return cp.multiply(projs, -1, out=projs)  # sunny side up

        if self.preload:
            if self.verbose:
                print("Preload requested. Loading all projections...")
            # get everything to RAM, avoids slow file loading
            loaded_projs = reflex.projs(self.projs_path, self.proj_ids,
                                        verbose=True)

            def _load_fn(ids):
                ids_lst = list(self.proj_ids)
                return np.asarray(
                    [loaded_projs[ids_lst.index(i)] for i in ids])
        else:
            def _load_fn(ids):
                return reflex.projs(self.projs_path, ids, verbose=True)

        # set up the buffer reconstruction
        self.bufferbp = ProjectionBufferBp(
            self.proj_ids,
            self.geometries,
            load_fn=_load_fn,
            preproc_fn=_preprocess_projs,
            batch_load=0,
            verbose=True,
            kernel_voxels_z=min(self.voxels[-1], 6),
            texture_type=self.texture_type)

        # preload also means prefilter
        if self.preload:
            self.bufferbp._load_proj(self.proj_ids)

        # start logging
        self._iteration = -1
        print(f"Clock start ticking with __iter__. It is now {time.time()}. ",
              end='')
        if self.start_time is None:
            print("I will start right away.")
            self.start_time = time.time()
        else:
            print(f"I will start at {self.start_time}, which is in "
                  f"{self.start_time - time.time()} seconds.")
            if self.start_time < time.time():
                warnings.warn("Start time is in the past. Provide a moment"
                              " in the future that allows sufficient time"
                              " for the dataset to load.")
            else:
                from threading import current_thread
                print(f"Experiment not started yet, sleeping "
                      f"{self.start_time - time.time()} seconds in "
                      f"thread {current_thread()}.")
                time.sleep(self.start_time - time.time())

        return self

    @property
    def time(self):
        return time.time() - self.start_time

    @property
    def iteration(self):
        return self._iteration

    def __next__(self):
        """Acquire a sample from the buffer"""

        self._iteration += 1
        # self.pre_reconstruction(self.bufferbp)

        t = self.time  # use the same time while calling the sampler
        # if t < 0:
        #     from threading import current_thread
        #     print(f"Experiment not started yet, sleeping {-t} seconds in"
        #           f" thread {current_thread()}.")
        #     time.sleep(-t)

        num = self.number(self.iteration, t)
        pos, rot = self.space(self.iteration, num, t)
        return self.reconstruct(
            self.bufferbp,
            num,
            self.reco_interval,
            pos,
            rot)

    def reconstruct(self,
                    bufferbp: ProjectionBufferBp,
                    num_start,
                    numbers_sample_size,
                    vol_position,
                    vol_rotation):
        """Make a Noise2Inverse reconstruction from the buffer."""

        # shift volume extent to position
        vol_extent_min = np.array(self.vol_ext_min) - vol_position
        vol_extent_max = np.array(self.vol_ext_max) - vol_position
        vol_geom = VolumeGeometry(self.voxels, [self.voxel_size] * 3,
                                  vol_extent_min, vol_extent_max,
                                  vol_rotation)

        if self.n2i_mode == 'even_odd' or self.n2i_mode == 'even_odd_mixed':
            # the following split of loops is necessary to facilitate a check
            # to make sure that there is never overlap between even/odd recons
            # so that it is impossible from noise information to spill over
            # between input and target
            in_ranges = []
            out_ranges = []
            if self.n2i_mode == 'even_odd_mixed':
                swap = np.random.choice([True, False])

            for t in range(self.seq_len):
                t_start = num_start + t * self.seq_stride
                t_stop = t_start + numbers_sample_size
                in_range = range(t_start, t_stop, 2)
                out_range = range(t_start + 1, t_stop + 1, 2)
                # swap when required by randomness, or to make sure that
                # even is always even and odd is always odd.
                if ((self.n2i_mode == 'even_odd_mixed' and swap)
                    or t_start % 2 == 1):
                    in_range, out_range = out_range, in_range

                in_ranges.append(in_range)
                out_ranges.append(out_range)

            for e in in_ranges:
                for nr in e:
                    for o in out_ranges:
                        if nr in o:
                            raise Exception("Error: projections from even"
                                            "recons are accidentally spilling "
                                            "over in odd reconstructions.")
            inputs = []
            targets = []
            with cp.cuda.Stream():
                for e, o in zip(in_ranges, out_ranges):
                    inputs.append(bufferbp(e, vol_geom))
                    targets.append(bufferbp(o, vol_geom))
            # ids = []
            # for e, o in zip(in_ranges, out_ranges):
            #     ids.append(e)
            #     ids.append(o)
            # out = bufferbp(ids, vol_geom)
            # inputs, targets = [], []
            # for i, rec in enumerate(out):
            #     if i % 2 == 0:
            #         inputs.append(rec)
            #     else:
            #         targets.append(rec)
        elif self.n2i_mode == 'full':
            with cp.cuda.Stream():
                in_ranges = []
                for t in range(self.seq_len):
                    t_start = num_start + t * self.seq_stride
                    t_stop = t_start + numbers_sample_size
                    in_range = range(t_start, t_stop)
                    in_ranges.append(in_range)

                inputs = []
                targets = []
                out = bufferbp(in_ranges, vol_geom)
                for rec in out:
                    inputs.append(rec)
                    targets.append(rec)

        else:
            raise ValueError(
                "Mode `self.n2i_mode` unknown. Choose 'even_odd', "
                "'even_odd_mixed', or 'full'.")

        info = {
            'vol_rotation': np.array(vol_rotation, dtype=np.float32),
            'vol_position': np.array(vol_extent_max, dtype=np.float32)
                            - np.array(vol_extent_min, dtype=np.float32),
            'voxel_size': np.array(self.voxel_size, dtype=np.float32),
            'num_start': np.array(num_start, dtype=np.int32)}
        # if self.device_ids is None or len(self.device_ids) == 0:
        #     out = ([torch.as_tensor(i, device='cuda') for i in inputs],
        #            [torch.as_tensor(t, device='cuda') for t in targets],
        #            info)
        # else:
        #     out = ([torch.asarray(i) for i in inputs],
        #            [torch.asarray(t) for t in targets],
        #            info)
        out = (
            torch.asarray(cp.asarray(inputs)),
            torch.asarray(cp.asarray(targets)),
            info)
        if self.squeeze:
            out = torch.squeeze(out[0]), torch.squeeze(out[1]), info
        return out

    @staticmethod
    def reflex_geom(settings, angles, corrections: Any = True, verbose=None):
        motor_geom = reflex.motor_geometry(
            settings,
            corrections,
            verbose=verbose)
        geom = reflex.circular_geometry(
            settings,
            initial_geom=motor_geom,
            angles=angles)
        geoms = []
        for i, (t, static_geom) in enumerate(geom.to_dict().items()):
            g = centralize(static_geom)
            hv = g.stage_rotation_matrix @ [0, 1, 0]
            vv = g.stage_rotation_matrix @ [0, 0, 1]
            g = reflex.centralize(static_geom)
            det = Detector(
                rows=g.detector.rows,
                cols=g.detector.cols,
                pixel_width=g.detector.pixel_width,
                pixel_height=g.detector.pixel_height)
            geom = ProjectionGeometry(
                source_pos=g.tube_position,
                det_pos=g.detector_position,
                u_unit=hv,
                v_unit=vv,
                detector=det)
            geoms.append(geom)

        return geoms

    def space(self, iteration, num, t):
        if isinstance(self.pos_min, Callable):
            pos_min = self.pos_min(iteration, num, t)
        else:
            pos_min = self.pos_min

        if isinstance(self.pos_max, Callable):
            pos_max = self.pos_max(iteration, num, t)
        else:
            pos_max = self.pos_max

        if isinstance(self.rot_min, Callable):
            rot_min = self.rot_min(iteration, num, t)
        else:
            rot_min = self.rot_min

        if isinstance(self.rot_max, Callable):
            rot_max = self.rot_max(iteration, num, t)
        else:
            rot_max = self.rot_max

        return self.space_sampler(
            vol_position_min=pos_min,
            vol_position_max=pos_max,
            vol_rotation_min=rot_min,
            vol_rotation_max=rot_max)

    @staticmethod
    def space_sampler(vol_rotation_min: Sequence, vol_rotation_max: Sequence,
                      vol_position_min: Sequence, vol_position_max: Sequence):
        # random point position
        space = np.subtract(vol_position_max, vol_position_min)
        assert np.all(space >= 0)
        pos = np.random.random_sample(3) * space + vol_position_min

        # random rotation parameter vector
        vol_rotation_min = np.array(vol_rotation_min)
        vol_rotation_max = np.array(vol_rotation_max)
        rot = vol_rotation_min + (
            np.random.random_sample(3) * (
            vol_rotation_max - vol_rotation_min))
        return pos, rot


class UniformTabletDataset(TabletDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def number(self, iteration, time):
        """Samples from `projs_ids` uniformly."""

        # minimum
        low = self.proj_ids.start
        high = (self.proj_ids.stop
                - self.reco_interval
                - (self.seq_len - 1) * self.seq_stride
                - 1)  # -1 because even/odd

        if not high > low:
            raise SampleIntervalTooSmallError(
                "Interval `proj_ids` needs to be sufficiently large to sample "
                "from (note the interval has to be large in the spatio-"
                "temporal case.")

        # options = []
        # for i in range(low, high):
        #     if i % self.reco_interval == 0:
        #        options.append(i)
        #
        # return np.random.choice(options)

        # note: `high` in randint() is exclusive
        return np.random.randint(low=low, high=high)


class ContinuousTabletDataset(TabletDataset):
    def __init__(self,
                 *args,
                 num_start: int = None,
                 num_increment: int = 15,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if num_start is None:
            self._num = self.proj_ids.start
        else:
            assert num_start in self.proj_ids
            self._num = num_start

        self._num_increment = num_increment

    def number(self, iteration, time):
        return self.proj_ids.start + iteration * self._num_increment


class RealtimeTabletDataset(TabletDataset):
    """Samples are provided based on running clock."""

    def __init__(self,
                 *args,
                 time_per_proj: float = 0.012,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.time_per_proj = time_per_proj
        self.reco_size = (
            self.reco_interval  # a single reco
            + (self.seq_len - 1) * self.seq_stride  # + extra frames
            + 1)  # + even-odd

    def number(self, iteration, t):
        # TODO: The dataloader is fetching one item after __iter__
        #     and the clock starts ticking directly
        # if self._time_start is None or iteration < 1:
        #     self._time_start = time.time() - self.time_per_proj * self.reco_interval

        # The experiment must have been running for sufficient time to have
        # filled the buffer, otherwise we cannot make reconstructions.
        # required_time = t - self.time_per_proj * self.reco_size
        # if required_time < 0:
        #     print(f"t (time since start): {t}")
        #     print(f"Tpp: {self.time_per_proj}, total time {self.time_per_proj * self.reco_size}")
        #     print(f"Time to start: {-required_time} seconds. Sleeping...")
        #     time.sleep(-required_time)
        # else:
        # if not self._ignore_late:
        #     warnings.warn(
        #         f"Dataset is initialized {required_time} past the given "
        #         f" starting time, which was {self._time_start}. "
        #         f"We could start giving out late data, but won't do it "
        #         f"unless you indicate this with `ignore_late`.")

        curr_num = self.proj_ids.start + int(t // self.time_per_proj)
        print(f"Current time: {t}. Current num: {curr_num}.")

        # remove unnecessary projections from the buffer
        # self.bufferbp.update_buffer = range(  # for memory management
        #     max(curr_num, self.proj_ids.start),
        #     curr_num + self.reco_size)
        #
        if curr_num >= self.proj_ids.stop:
            raise StopIteration

        return curr_num


def realtime(
    path,
    settings_path,
    proj_ids,
    reco_interval,
    time_per_proj,
    device_ids=(1,),
    start_time: float = None,
    voxels: Any = 100,
    **kwargs):
    # TODO: this is hardcoded for the tablet scan
    # Note that the time_per_proj cannot be accurately inferred from the
    # settings file and need to be supplied by the user. I calculated this
    # as number of proj. files / number of seconds
    #       5 min / 25000 projs = 0.012
    # alternatively: the rot_obj speed in some settings files seem to be
    # about 200 degrees / second. So the time per projection can also be
    # inferred from the number of projections per 360 degrees (which is 150).
    #       200 / 360 * 150 = 0.012
    assert 'preload' not in kwargs.keys() or kwargs['preload'] is True, (
        "Preloading to RAM must be enabled for real-time datasets, because filesystem loading "
        " is slowing down significantly (especially for mounted drives/NFS.)")

    return RealtimeTabletDataset(
        start_time=start_time,
        settings_path=settings_path,
        projs_path=path,
        darks_path=settings_path,
        flats_path=settings_path,
        proj_ids_lim=proj_ids,
        voxels=voxels,
        nr_angles_per_rot=150,
        reco_interval=reco_interval,
        verbose=True,
        device_ids=device_ids,
        # time_per_proj=0.012,  # TODO: see above
        time_per_proj=time_per_proj,  # TODO: see above
        preload=True,
        **kwargs)


def _worker_init_fn(worker_id):
    """Sets CuPy's device to one of the free devices."""

    # seed is different everytime the function is launched, which is good
    # because otherwise we would be reiterating the same samples
    worker_seed = torch.initial_seed() % 2 ** 32
    print(f"Worker {worker_id} seed: {worker_seed}")

    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # TODO: how to get the "next free" device? and remove `device_ids`
    #   from dataset.
    worker_info = torch.utils.data.get_worker_info()
    dev = worker_info.dataset.device_ids[worker_id]
    # dev = cp.cuda.get_device_id()
    cp.cuda.runtime.setDevice(dev)


def dataloader_from_dataset(ds, batch_size, generator_seed=0):
    assert hasattr(ds, 'device_ids'), \
        "The dataset needs to have a custom" \
        " property `device_ids`."
    g = torch.Generator()
    g.manual_seed(generator_seed)
    # assert len(ds.device_ids) > 0
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0 if ds.device_ids is None else len(ds.device_ids),
        worker_init_fn=_worker_init_fn,
        pin_memory=True,
        generator=g)
    return dl


def reconstruct_3d(
    path,
    settings_path,
    proj_ids,
    nr_angles_per_rot=150,
    reconstruction_rotation_intval=2 * np.pi,
    plot=True,
    save=False,
    filter='ramlak',
    t=400,
    d=None,
    algo='fdk'
):
    settings = reflex.Settings.from_path(settings_path)
    if nr_angles_per_rot is None:
        nr_angles_per_rot = len(proj_ids)

    angles = (np.array(proj_ids)
              * reconstruction_rotation_intval / nr_angles_per_rot)

    darks = reflex.darks(settings_path)
    whites = reflex.flats(settings_path)
    dark = None
    if len(darks) != 0:
        if len(darks) > 1:
            darks = darks.mean(0)
        dark = cp.squeeze(cp.array(darks))

    white = None
    if len(whites) != 0:
        if len(whites) > 1:
            whites = (whites - darks if len(darks) != 0 else whites).mean(0)
        white = cp.squeeze(cp.array(whites))

    def _preproc(projs):
        xp = cp.get_array_module(projs[0])
        for p in projs:
            if dark is not None:
                xp.subtract(p, dark, out=p)  # remove darkfield
            if white is not None:
                xp.divide(p, white, out=p)
            xp.log(p, out=p)  # take -log to linearize detector values
            xp.multiply(p, -1, out=p)  # sunny side up

    geometry = TabletDataset.reflex_geom(settings, angles, True)
    projections = reflex.projs(path, proj_ids)

    if algo == 'fdk':
        vol = astrapy.fdk(
            projections,
            geometry,
            (t, t, None),
            (-d, -d, -d) if d is not None else [None] * 3,
            (d, d, d) if d is not None else [None] * 3,
            preproc_fn=_preproc,
            filter=filter,
            verbose=True)
    elif algo == 'sirt':
        vol = astrapy.sirt_experimental(
            projections,
            geometry,
            preproc_fn=_preproc,
            volume_shape=(t, t, None),
            volume_extent_min=(-d, -d, -d) if d is not None else [None] * 3,
            volume_extent_max=(d, d, d) if d is not None else [None] * 3,
        )
    else:
        raise ValueError

    if plot:
        import pyqtgraph as pq
        pq.image(vol)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(vol[..., vol.shape[2] // 2])
        plt.show()

    if save:
        infodict = {
            "timeframe": proj_ids[0],
            "volume": vol,
            "algorithm": algo,
        }
        fname = f'out_{algo}_{proj_ids[0]}.npy'
        # np.save(fname,
        #         infodict,
        #         allow_pickle=True)
        np.save(fname, vol, allow_pickle=True)
        print(f"Saved to {fname}.")

    return vol
