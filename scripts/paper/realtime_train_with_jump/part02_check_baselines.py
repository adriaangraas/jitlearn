import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from config import *

from jitlearn.tablet import ContinuousTabletDataset, UniformTabletDataset, \
    dataloader_from_dataset
from jitlearn.training import find_checkpoints, restore


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    fnames = find_checkpoints('.', "baseline_before_jump_r2_batch(\\d+)\.ckpt", )
    ckpts = {batch: restore(model, fname)
             for batch, fname in fnames.items()}

    accuracy = []
    for batch, ckpt in ckpts.items():
        accuracy.append(ckpt[1]['denoised'] / ckpt[1]['noisy'])

    # select best model
    # plt.figure()
    # plt.plot(accuracy)
    # plt.show()
    BEST_MODEL = 40000  # this is one is actually really good

    model = ckpts[100][0][0].cuda()

    ds = ContinuousTabletDataset(
        path,
        settings_path,
        range(PROJ_IDS.start, JUMP_AT),
        (250, 250, VOXELS_Z),
        (0., 0., 0.),
        (0., 0., 0.),
        ROT_MIN,
        ROT_MAX,
        sequence_length=SEQ_LEN,
        sequence_stride=SEQ_STRIDE,
        device_ids=[3],
        iso_voxel_size=ISO_VOXEL_SIZE,
        num_increment=15,
        preload=False
    )

    pairs = dataloader_from_dataset(ds, batch_size=1)
    pairs = iter(pairs)

    def next_item():
        train_item = next(pairs)
        input, target, info = train_item
        # input = torch.squeeze(input).detach().numpy()
        # target = torch.squeeze(target).detach().numpy()
        print(f"Current frame: {info['num_start'].numpy()[0]}")
        return input, target, info


    input, target, info = next_item()

    fig, axs = plt.subplots(1, 2)
    axs[0].set_title(f"Target")
    obj0 = axs[0].imshow(
        torch.squeeze(target).detach().numpy()[0],
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')
    axs[1].set_title(f"Outcome")
    obj1 = axs[1].imshow(
        torch.squeeze(target).detach().numpy()[0],
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')

    def update(i):
        input, target, info = next_item()
        out = model(input.cuda())
        obj0.set_array(torch.squeeze(target).detach().numpy()[0])
        obj1.set_array(torch.squeeze(out.detach()).cpu().numpy())
        return obj0, obj1

    _ = animation.FuncAnimation(fig, update, blit=True)
    plt.show()