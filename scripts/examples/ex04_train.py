from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from time import time

from mpl_toolkits.axes_grid1 import ImageGrid
from torch import optim

from jitlearn import BufferingMultiProcessingDataLoaderIter
from jitlearn.tablet import RealtimeTabletDataset, dataloader_from_dataset
from unet import Unet3d


def train(model, opt, train_pairs, eval_pairs):
    fig, axs = plt.subplots(1, 2)
    movie, target, _ = next(eval_pairs)
    _ = next(eval_pairs)
    out = movie.detach().cpu().numpy()
    middle_frame = (movie.shape[1] - 1) // 2
    obj0 = axs[0].imshow(
        out[0, middle_frame],
        interpolation=None,
        vmin=0.0, vmax=0.75,# / 2,
        aspect='equal')
    obj1 = axs[1].imshow(
        out[0, -1],
        interpolation=None,
        vmin=0.0, vmax=0.75,# / 2,
        aspect='equal')

    fig2 = plt.figure()
    movie, target, _ = next(train_pairs)
    grid = ImageGrid(fig2, 111, nrows_ncols=(4, 4), axes_pad=0.1)
    objs = []
    for ax, im in zip(grid, movie[:16]):
        objs.append(ax.imshow(im[0, ...], interpolation=None,
                              vmin=0.0, vmax=0.75))
    for ax, obj in zip(grid, objs):
        ax.draw_artist(obj)

    def update(i):
        eval_item = next(eval_pairs)
        movie, target, info = eval_item
        model.eval()
        out = model(movie.cuda())
        out = out.detach().cpu().numpy()
        obj0.set_array(target.detach().cpu().numpy()[0, -1])
        obj1.set_array(out[0, 0])
        return [obj0, obj1]

    def update2(i):
        for j in range(20):
            print('.', end='')
            train_item = next(train_pairs)
            movie, target, info = train_item
            model.train()
            opt.zero_grad()
            out = model(movie.cuda())
            loss = model.crit(out, target.cuda())
            loss.backward()
            opt.step()

        for ax, obj, im in zip(grid, objs, movie[:16]):
            obj.set_data(im[0, ...])
            ax.draw_artist(obj)

    anim = FuncAnimation(fig, update)
    anim2 = FuncAnimation(fig2, update2)
    plt.show()


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)

    SEQ_LEN = 1
    SEQ_STRIDE = 60
    VOXELS_Z = 5
    PROJ_IDS = range(1000, 13000)
    RECO_INTERVAL = 150
    VOXEL_SIZE = 0.25
    # vertical slice around rotational axis
    # ROT_MIN = (np.pi, -.5 * np.pi, 0)
    # ROT_MAX = (np.pi, -.5 * np.pi, 2 * np.pi)
    # around vertical axis and one-way flip around one of the horiz axis
    ROT_MIN = (0, 0., np.pi)
    ROT_MAX = (0, 0., np.pi)
    TIME_PER_PROJ = 0.012 * 5
    TRAIN_MAIN_DIR = 'handsoap'
    TRAIN_SUB_DIR = 'scan_1'
    EVAL_MAIN_DIR = 'handsoap'
    EVAL_SUB_DIR = 'scan_1'
    START_TIME = time() + 60
    basedir = "/export/scratch2/adriaan/zond_fastdvdnet_training"
    # TEMPLATE_PATH = "/bigstore/felix/CT/TabletDissolve2/12ms4x4bin/{main_dir}/{sub_dir}"
    TEMPLATE_PATH = "/export/scratch3/adriaan/bigstore/{main_dir}/{sub_dir}"

    train_path = TEMPLATE_PATH.format(main_dir=TRAIN_MAIN_DIR,
                                      sub_dir=TRAIN_SUB_DIR)
    eval_path = TEMPLATE_PATH.format(main_dir=EVAL_MAIN_DIR,
                                     sub_dir=EVAL_SUB_DIR)
    train_settings_path = TEMPLATE_PATH.format(main_dir=TRAIN_MAIN_DIR,
                                               sub_dir='pre_scan')
    eval_settings_path = TEMPLATE_PATH.format(main_dir=EVAL_MAIN_DIR,
                                              sub_dir='pre_scan')

    # Makes sure that the patch is contained only in the ROI
    PATCH_SIZE = 60
    padding = VOXEL_SIZE * PATCH_SIZE / 2
    posmin = (-25 + padding, -25 + padding, -25 + padding)
    posmax = (25 - padding, 25 - padding, 25 - padding)
    posmin = (-25 + padding, -25 + padding, 0)
    posmax = (25 - padding, 25 - padding, 0)
    # posmin = (0, 0, 0)
    # posmax = (0, 0, 0)

    # If patch is too big, padding will push the possible positions over
    # the center of the volume. In this case we won't be taking random
    # patches but just the central position (0., 0., 0.) for each axis
    # that is going over the center.
    posmin = [np.min((0, s)) for s in posmin]
    posmax = [np.max((0, s)) for s in posmax]

    train_ds = RealtimeTabletDataset(
        train_path,
        train_settings_path,
        PROJ_IDS,
        (PATCH_SIZE, PATCH_SIZE, 1),
        posmin,
        posmax,
        (0., 0., 0.),
        (0., 0., 0.),
        time_per_proj=TIME_PER_PROJ,
        start_time=START_TIME,  # start train and eval at same time
        sequence_length=SEQ_LEN,
        sequence_stride=SEQ_STRIDE,
        device_ids=[1],
        iso_voxel_size=VOXEL_SIZE,
        preload=True)
    train_dl = dataloader_from_dataset(train_ds, batch_size=32)

    eval_ds = RealtimeTabletDataset(
        eval_path,
        eval_settings_path,
        PROJ_IDS,
        (200, 200, 1),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        time_per_proj=TIME_PER_PROJ,
        start_time=START_TIME,  # start train and eval at same time
        sequence_length=SEQ_LEN,
        sequence_stride=SEQ_STRIDE,
        device_ids=[2],
        iso_voxel_size=VOXEL_SIZE,
        preload=True,
        # n2i_mode='full'
    )
    eval_dl = dataloader_from_dataset(eval_ds, batch_size=1)

    # These datasets start when __iter__ is called
    train_pairs = iter(BufferingMultiProcessingDataLoaderIter(
        train_dl, max_len=2000))
    eval_pairs = iter(BufferingMultiProcessingDataLoaderIter(eval_dl, 1))

    model = Unet3d(
        space_dims=3 if VOXELS_Z > 1 else 2,
        sequence_length=SEQ_LEN,
        attack_frame=(SEQ_LEN - 1) // 2,
        with_temporal_downsampling=False,
        initial_down_channels=16,
        verbose=True,
        skip='concat',
        nr_levels=3,
        superskip=.0)

    train(model.cuda(),
          optim.Adam(model.parameters(), lr=5e-5),
          train_pairs, eval_pairs)