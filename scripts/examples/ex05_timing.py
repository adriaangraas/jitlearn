import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from jitlearn.tablet import UniformTabletDataset


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    MAIN_DIR = 'handsoap'
    SUB_DIR = 'scan_1'
    TEMPLATE_PATH = "/export/scratch3/adriaan/bigstore/{main_dir}/{sub_dir}"
    path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir=SUB_DIR)
    settings_path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir='pre_scan')

    ds = UniformTabletDataset(
        path,
        settings_path,
        range(1000, 2000),
        (50, 50, 1),  # a 2000x2000 slice
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        device_ids=[1],
        iso_voxel_size=0.25,
        preload=True,
    )
    ds = iter(ds)

    for _ in tqdm(range(200)):  # warmup
        next(ds)

    fig, axs = plt.subplots(1, 3)
    data = []
    means = []
    stddevs = []
    for i in tqdm(range(1001)):
        plt.cla()
        start = time.time()
        _ = next(ds)
        elapsed = time.time() - start
        data.append(elapsed)
        means.append(np.array(data).mean())
        stddevs.append(np.std(np.array(data)))

        if i % 50 == 0:
            print(f"{data[-1]} {means[-1]} {stddevs[-1]}")
            axs[0].set_title(f"Data")
            line0 = axs[0].plot(range(len(data)), data)[0]
            axs[1].set_title(f"Mean")
            line1 = axs[1].plot(range(len(means)), means)[0]
            axs[2].set_title(f"Std.dev.")
            line2 = axs[2].plot(range(len(stddevs)), stddevs)[0]
            plt.pause(.01)

    plt.show()
