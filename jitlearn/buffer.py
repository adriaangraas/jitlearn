import time
from typing import Any, Callable, List, Sequence

import cupy as cp
import numpy as np
import torch
from astrapy import aspitched, copy_to_texture, process
from astrapy.kernels import ConeBackprojection, GeometrySequence
from torch.utils.data.dataloader import (_DatasetKind,
                                         _MultiProcessingDataLoaderIter)
from torch.utils.data.dataloader import _utils


class ProjectionBufferBp:
    """Reconstructions from a GPU buffer"""

    def __init__(
        self,
        projection_ids: Sequence,
        geometries: dict,
        load_fn: Callable,
        filter: str = 'ramlak',
        preproc_fn: Callable = None,
        texture_type: str = 'pitch2D',
        batch_load: int = 0,
        verbose: bool = False,
        kernel_voxels_z: int = 1):
        """

        :param projection_ids: The projection ids that are in the
            buffer. Note that we often lazy-load these projections, so they
            may not be in memory.
        :param geometries: Geometries of all potential ids. These are not
            lazily generated.
        :param load_fn: a function to get projections (from disk)
        :param filter: FDK filter
        :param preproc_fn: a function to preprocess raw projections on GPU
        :param texture_type:
        :param batch_load: if >0, on preprocessing extra projections are taken.
            This may be useful with, i.e., a moving buffer in time.
        :param verbose:
        """
        self.verbose = verbose
        self._texture_type = texture_type
        self._projection_ids = projection_ids
        self._geoms = geometries
        self._filter = filter
        self._preproc_fn = preproc_fn
        self._load_fn = load_fn
        self._textures = {}
        self._batch_load = batch_load
        self._kernel = ConeBackprojection(
            voxels_per_block=(16, 32, kernel_voxels_z),
            min_limit_projs=75)
        sorted(geometries)

        # this is to support taking contiguous slices which increases memory usage
        # but is much faster than copying
        geoms = [v for k, v in geometries.items()]
        geoms_even = [v for k, v in geometries.items() if k % 2 == 0]
        geoms_odd = [v for k, v in geometries.items() if k % 2 == 1]
        self._geomseq = GeometrySequence.fromList(geoms)
        self._geomseq_even = GeometrySequence.fromList(geoms_even)
        self._geomseq_odd = GeometrySequence.fromList(geoms_odd)
        self._geomseq_id = {
            id: i for i, id in enumerate(geometries.keys())}
        self._geomseq_id_mixed = {
            id: int(i / 2) for i, id in enumerate(geometries.keys())}

    def _load_proj(self, ids):
        """
        Load and preprocesses `ids` but upload a minimum of `self._min_load`

        :param ids:
        :return:
        """
        # do not load the already loaded ids
        ids_to_load = list(
            filter(lambda i: i not in self._textures.keys(), ids))

        if not len(ids_to_load) == 0:
            if len(ids_to_load) < self._batch_load:
                # find ourselves `_min_load` unloaded projection ids
                for i in self._projection_ids:
                    if i not in self._textures.keys() and i not in ids_to_load:
                        ids_to_load.append(i)
                    if len(ids_to_load) == self._batch_load:
                        break

            # load all the required ids (+ eventual extra in the batch)
            projections_cpu = self._load_fn(ids_to_load)

            # projs need to be independent 2D arrays in order to be pitched
            projs_gpu = cp.asarray(projections_cpu, dtype=cp.float32)
            if self._preproc_fn is not None:
                self._preproc_fn(projs_gpu)
            # preweighting requires geometries to be given at __init__
            geoms = [self._geoms[id] for id in ids_to_load]
            process.preweight(projs_gpu, geoms)
            if self._filter:
                process.filter(projs_gpu, filter=self._filter,
                               verbose=self.verbose)

            for i, p in zip(ids_to_load, projs_gpu):
                if self._texture_type.lower() == 'pitch2d':
                    p = aspitched(cp.array(p, copy=False))
                self._textures[i] = copy_to_texture(p, type=self._texture_type)

        return [self._textures[i] for i in ids]

    def _clear_proj(self):
        for key in list(self._textures.keys()):
            if key not in self._projection_ids:
                self._textures.pop(key)
        # doesn't seem to be necessary:
        # cp.get_default_memory_pool().free_all_blocks()

    @property
    def update_buffer(self):
        return self._projection_ids

    @update_buffer.setter
    def update_buffer(self, ids):
        self._projection_ids = ids
        self._clear_proj()

    def __call__(self,
                 ids: range,
                 volume_shape: Sequence,
                 volume_extent_min: Sequence,
                 volume_extent_max: Sequence,
                 volume_rotation: Sequence,
                 dtype=cp.float32
                 ):
        """Reconstruct a volume."""
        assert ids.step in (1, 2)
        if not all(i in self._projection_ids for i in ids):
            raise ValueError(
                f"`ids` have to be indices of the projections. Some of "
                f" {ids} are not in {self._projection_ids}.")

        textures = self._load_proj(ids)

        if not hasattr(self, 'volume'):
            self.volume = cp.zeros(tuple(volume_shape), dtype=dtype)
        else:
            self.volume.fill(0.)

        if ids.step == 1:
            idx1 = self._geomseq_id[ids.start]
            idx2 = self._geomseq_id[ids.stop]
            sl = slice(idx1, idx2)
            subgeomseq = self._geomseq.take(sl)
        elif ids.step == 2:
            idx1 = self._geomseq_id_mixed[ids.start]
            idx2 = self._geomseq_id_mixed[ids.stop]
            if ids.start % 2 == 0:
                subgeomseq = self._geomseq_even.take(slice(idx1, idx2))
            else:
                subgeomseq = self._geomseq_odd.take(slice(idx1, idx2))


        params = self._kernel.geoms2params(
            subgeomseq,
            self.volume.shape,
            volume_extent_min,
            volume_extent_max,
            volume_rotation,
            fdk_weighting=False)

        self._kernel(
            textures,
            params,
            self.volume,
            tuple(volume_extent_min),
            tuple(volume_extent_max),
            tuple(volume_rotation))

        self.volume[...] = cp.reshape(self.volume,
                                      tuple(reversed(self.volume.shape))).T
        return cp.copy(self.volume)


class BufferingMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    """This custom multiprocessing iterator does the following:
        - it never waits for task to be finished, so data loading is
          continuous
        - all data is stored in a buffer. If no new sample is available,
          it just retrieves something from the buffer
        - `max_len` defines the size of the buffer. If the buffer is full
          the start of the buffer is pruned.
    This cannot be passed to the dataloader _iterator argument, because we
    require `loader` in the __init__.
    Unfortunately I don't know how to use the batch assambly/disassamble
    PyTorch tools, so I'm hacking it here.
    """

    def __init__(self, loader, max_len=100, debug=False):
        super(BufferingMultiProcessingDataLoaderIter, self).__init__(loader)
        self._batch_size = loader.batch_size
        # a list-based queue, no performance loss since buffer is small
        self._buffer = []
        self._max_len = max_len

        self._debug = debug
        if self._debug:
            self._times = []

        if self._batch_size > self._max_len:  # Adriaan
            raise Exception(
                "Cannot have a buffer size that is smaller than the number of items"
                " in a batch.")

    def _contribute_buffer(self, data):
        for i in range(self._batch_size):
            if self._debug:
                self._times += [time.time()]

            # TODO: hacky way, how to decompose batches, and recompose again
            #     later in a generic way?
            self._buffer += [[
                data[0][i],
                data[1][i],
                {k: elem[i] for k, elem in data[2].items()}
            ]]

        buffer_len = len(self._buffer)
        if buffer_len > self._max_len:
            for _ in range(buffer_len - self._max_len):
                self._buffer.pop(0)  # fast enough for small buffers

            if self._debug:
                for _ in range(buffer_len - self._max_len):
                    self._times.pop(0)

        if self._debug:
            print(f"First time: {self._times[0]}")
            print(f"Last time: {self._times[-1]}")
            print(f"Nr. elements: {len(self._times)}")
            if len(self._times) > 1:
                reco_speed = (
                    len(self._times) / (self._times[-1] - self._times[0]))
                print(f"Reco speed: {reco_speed}")

    def _get_buffer(self, inds=None):
        if len(self._buffer) < self._batch_size:
            return False
        if inds is None:
            inds = np.random.choice(len(self._buffer), self._batch_size,
                                    replace=False)
        elif inds == 'last':
            inds = [-i - 1 for i in range(self._batch_size)]

        return [
            torch.concat(
                [torch.unsqueeze(self._buffer[i][0], dim=0) for i in inds]),
            torch.concat(
                [torch.unsqueeze(self._buffer[i][1], dim=0) for i in inds]),
            {k: torch.concat([
                torch.unsqueeze(self._buffer[i][2][k], dim=0) for i in inds])
                for k in iter(self._buffer[0][2].keys())}
        ]

    def last(self):
        return self._get_buffer('last')

    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[
                    worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            # Adriaan: I think this is what happens if an out-of-order
            #   sample is gotten, see below.
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                pdata = self._process_data(data)
                self._contribute_buffer(pdata)  # Adriaan
                buffer_data = self._get_buffer()
                if buffer_data is not False:
                    return buffer_data

            assert not self._shutdown and self._tasks_outstanding > 0
            # idx, data = self._get_data()  # Adriaan: old
            success, data_or_none = self._try_get_data(
                timeout=0.)  # Adriaan: new
            if success:  # Adriaan
                idx, data = data_or_none
                self._tasks_outstanding -= 1
                if self._dataset_kind == _DatasetKind.Iterable:
                    # Check for _IterableDatasetStopIteration
                    if isinstance(data,
                                  _utils.worker._IterableDatasetStopIteration):
                        if self._persistent_workers:
                            self._workers_status[data.worker_id] = False
                        else:
                            self._mark_worker_as_unavailable(data.worker_id)
                        self._try_put_index()
                        continue

                if idx != self._rcvd_idx:
                    # store out-of-order samples
                    self._task_info[idx] += (data,)
                else:
                    del self._task_info[idx]
                    pdata = self._process_data(data)
                    self._contribute_buffer(pdata)  # Adriaan
                    buffer_data = self._get_buffer()
                    if buffer_data is not False:
                        return buffer_data
            else:  # Adriaan
                buffer_data = self._get_buffer()
                if buffer_data is not False:
                    return buffer_data
