from time import sleep, time

import matplotlib.pyplot as plt
import torch
from matplotlib import animation

from jitlearn import BufferingMultiProcessingDataLoaderIter
from jitlearn.tablet import dataloader_from_dataset, RealtimeTabletDataset

if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    MAIN_DIR = 'handsoap'
    SUB_DIR = 'scan_1'
    TEMPLATE_PATH = "/export/scratch3/adriaan/bigstore/{main_dir}/{sub_dir}"
    path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir=SUB_DIR)
    settings_path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir='pre_scan')

    ds = RealtimeTabletDataset(
        path,
        settings_path,
        range(1000, 4000),
        (200, 200, 3),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        time_per_proj=0.012 * 10,
        start_time=time() + 10,
        device_ids=[0],
        iso_voxel_size=0.25,
        preload=True)
    dl = dataloader_from_dataset(ds, batch_size=1)

    # note: for a realtime dataset, use our custom iterator to continuously
    # generate samples, and specify the size of the buffer
    pairs = iter(BufferingMultiProcessingDataLoaderIter(dl, 10))


    def next_item():
        train_item = next(pairs)
        input, target, info = train_item
        input = torch.squeeze(input).detach().numpy()
        target = torch.squeeze(target).detach().numpy()
        return input, target, info

    input, target, info = next_item()

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title(f"Input")
    obj0 = axs[0].imshow(
        input,
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')
    axs[1].set_title(f"Target")
    obj1 = axs[1].imshow(
        target,
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')

    def update(i):
        input, target, info = next_item()
        print(f"Current frame: {info['num_start'].numpy()[0]}")
        obj0.set_array(input)
        obj1.set_array(target)
        return obj0, obj1

    _ = animation.FuncAnimation(fig, update, blit=True)
    plt.show()