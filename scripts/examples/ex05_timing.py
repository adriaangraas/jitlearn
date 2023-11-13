import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import cupy as cp

from jitlearn.tablet import UniformTabletDataset


def astra_gen(voxels, nr_samples, nr_angles=75, detector_shape=(384, 261)):
    """
    This function mimicks an ideal ASTRA implementation, where pre-filtered
    data is already on the GPU using GPU Link, and where the data is already
    in a format that is suitable for Noise2Inverse without copying. I think it
    would be difficult to write this practically, because of the different
    axis convention.
    """
    import astra
    import astra.experimental

    angles = np.linspace(0, 2 * np.pi, nr_angles, False)
    vol_data_cpu = np.zeros(voxels, dtype=np.float32)
    vol_data_cpu[17:113, 17:113, 17:113] = 1
    vol_data_cpu[33:97, 33:97, 33:97] = 0
    vectors = np.zeros((len(angles), 12))
    for i in range(len(angles)):
        # source
        vectors[i, 0] = np.sin(angles[i]) * 1000
        vectors[i, 1] = -np.cos(angles[i]) * 1000
        vectors[i, 2] = 0
        # center of detector
        vectors[i, 3:6] = 0
        # vector from detector pixel (0,0) to (0,1)
        vectors[i, 6] = np.cos(angles[i])
        vectors[i, 7] = np.sin(angles[i])
        vectors[i, 8] = 0
        # vector from detector pixel (0,0) to (1,0)
        vectors[i, 9] = 0
        vectors[i, 10] = 0
        vectors[i, 11] = 1

    proj_data_cpu = np.random.random(
        (detector_shape[0], nr_angles, detector_shape[1]))
    vol_data = cp.zeros(  # to output into
        (vol_data_cpu.shape[2], vol_data_cpu.shape[0],
         vol_data_cpu.shape[1]), cp.float32)
    proj_data = cp.asarray(proj_data_cpu, dtype=cp.float32)
    z, y, x = vol_data.shape
    vol_link = astra.pythonutils.GPULink(vol_data.data.ptr, x, y, z, x * 4)
    proj_data = cp.zeros(proj_data.shape, cp.float32)
    z, y, x = proj_data.shape
    proj_link = astra.pythonutils.GPULink(proj_data.data.ptr, x, y, z, x * 4)
    vol_geom = astra.create_vol_geom(*voxels)

    while True:
        r = []
        for _ in range(2 * nr_samples):  # (input and target) * nr timesteps
            # note that `vectors` changes every iteration because of the random
            # sampling, so it has to be within this loop.
            proj_geom = astra.create_proj_geom('cone_vec', *detector_shape,
                                               vectors)
            proj_cfg = {'type': 'cuda3d',
                        'VolumeGeometry': vol_geom,
                        'ProjectionGeometry': proj_geom,
                        'options': {}}
            projector_id = astra.projector3d.create(proj_cfg)
            astra.experimental.direct_FPBP3D(projector_id,
                                             vol_link, proj_link, 1, "BP")
            # for a fair comparison
            out = torch.as_tensor(cp.copy(vol_data), device='cuda')
            r.append(out)

        yield r


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    MAIN_DIR = 'handsoap'
    SUB_DIR = 'scan_1'
    SEQ_LEN = 1
    VOXELS = (20, 20, 1)
    TEMPLATE_PATH = "/export/scratch3/adriaan/bigstore/{main_dir}/{sub_dir}"
    path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir=SUB_DIR)
    settings_path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir='pre_scan')
    WITH_ASTRA = False

    ds = UniformTabletDataset(
        path,
        settings_path,
        range(1000, 2000),
        VOXELS,
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        iso_voxel_size=0.25,
        sequence_length=SEQ_LEN,
        sequence_stride=60,
        preload=True,
        # texture_type='pitch2d'
    )
    ds = iter(ds)
    if WITH_ASTRA:
        astra_ds = astra_gen(VOXELS, SEQ_LEN)

    for _ in tqdm(range(20)):  # warmup
        next(ds)
        if WITH_ASTRA:
            next(astra_ds)

    fig, axs = plt.subplots(1, 3)
    data = []
    data_astra = []
    means = []
    means_astra = []
    stddevs = []
    stddevs_astra = []

    for i in tqdm(range(1001)):
        plt.cla()

        start = time.time()
        input, target, info = next(ds)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.time() - start
        data.append(elapsed)
        means.append(np.array(data).mean())
        stddevs.append(np.std(np.array(data)))

        # import pyqtgraph as pq
        # pq.image(input[0].detach().cpu().numpy())

        if WITH_ASTRA:
            start = time.time()
            _ = next(astra_ds)
            cp.cuda.get_current_stream().synchronize()
            elapsed = time.time() - start
            data_astra.append(elapsed)
            means_astra.append(np.array(data_astra).mean())
            stddevs_astra.append(np.std(np.array(data_astra)))

        if i % 20 == 0:
            print(f"{data[-1]} {means[-1]} {stddevs[-1]}")
            axs[0].set_title(f"Data")
            axs[0].plot(range(len(data)), data, color='green')
            if WITH_ASTRA:
                axs[0].plot(range(len(data_astra)), data_astra, color='red')
            axs[1].set_title(f"Mean")
            axs[1].plot(range(len(means)), means, color='green')
            if WITH_ASTRA:
                axs[1].plot(range(len(means_astra)), means_astra, color='red')
            axs[2].set_title(f"Std.dev.")
            axs[2].plot(range(len(stddevs)), stddevs, color='green')
            if WITH_ASTRA:
                axs[2].plot(range(len(stddevs_astra)), stddevs_astra, color='red')
            plt.pause(.01)

    plt.show()
