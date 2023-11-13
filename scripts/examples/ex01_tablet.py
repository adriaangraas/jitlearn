import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from jitlearn.tablet import ContinuousTabletDataset, UniformTabletDataset, \
    dataloader_from_dataset

if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    MAIN_DIR = 'handsoap'
    SUB_DIR = 'scan_1'
    TEMPLATE_PATH = "/export/scratch3/adriaan/bigstore/{main_dir}/{sub_dir}"
    path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir=SUB_DIR)
    settings_path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir='pre_scan')

    ds = ContinuousTabletDataset(
        path,
        settings_path,
        range(1000, 12000),
        (200, 200, 1),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        device_ids=[0],
        iso_voxel_size=0.25,
        num_increment=150,
    )

    # ds = UniformTabletDataset(
    #     path,
    #     settings_path,
    #     range(1000, 13000),
    #     (200, 200, 1),
    #     (0., 0., 0.),
    #     (0., 0., 0.),
    #     (0., 0., 0.),
    #     (0., 0., 0.),
    #     device_ids=[1],
    #     iso_voxel_size=0.25,
    # )

    # note: for a continuous dataset, select batch_size=1 to have a
    # time increment of `num_increment` per batch.
    pairs_train = dataloader_from_dataset(ds, batch_size=1)
    pairs_train = iter(pairs_train)


    def next_item():
        train_item = next(pairs_train)
        input, target, info = train_item
        input = input[0].detach().numpy()[0]
        target = target[0].detach().numpy()[0]
        print(f"Current frame: {info['num_start'].numpy()[0]}")
        return input, target, info

    input, target, info = next_item()

    fig, axs = plt.subplots(1, 3)
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
    axs[2].set_title(f"Difference")
    obj2 = axs[2].imshow(
        input - target,
        interpolation=None,
        vmin=-0.75/10, vmax=0.75/10,
        cmap='RdBu',
        aspect='equal')


    def update(i):
        input, target, info = next_item()
        obj0.set_array(input)
        obj1.set_array(target)
        obj2.set_array(input - target)
        return obj0, obj1, obj2


    _ = animation.FuncAnimation(fig, update, blit=True)
    plt.show()
