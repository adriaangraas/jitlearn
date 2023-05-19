import itertools
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import animation
from config import *
from jitlearn.tablet import UniformTabletDataset, dataloader_from_dataset


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    # training dataset
    train_ds = UniformTabletDataset(
        path,
        settings_path,
        range(PROJ_IDS.start + 1000),
        TRAIN_VOXELS,
        POS_MIN,
        POS_MAX,
        ROT_MIN,
        ROT_MAX,
        sequence_length=SEQ_LEN,
        sequence_stride=SEQ_STRIDE,
        device_ids=[1],
        iso_voxel_size=ISO_VOXEL_SIZE,
        preload=True
    )

    train_pairs = dataloader_from_dataset(train_ds, batch_size=BATCH_SIZE)
    train_pairs = iter(train_pairs)

    def next_item():
        train_item = next(train_pairs)
        input, target, info = train_item
        input = torch.squeeze(input).detach().numpy()
        target = torch.squeeze(target).detach().numpy()
        print(f"Current frame: {info['num_start'].numpy()[0]}")
        return input, target, info

    input, target, info = next_item()

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title(f"Input")
    obj0 = axs[0].imshow(
        input[0, 0],
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')
    axs[1].set_title(f"Target")
    obj1 = axs[1].imshow(
        target[0, 0],
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')

    def update(i):
        input, target, info = next_item()
        obj0.set_array(input[0, 0])
        obj1.set_array(target[0, 0])
        return obj0, obj1

    _ = animation.FuncAnimation(fig, update, blit=True)
    plt.show()
