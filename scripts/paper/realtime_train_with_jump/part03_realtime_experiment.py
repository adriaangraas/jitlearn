import itertools

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch import optim
from time import time
from config import *

from jitlearn import BufferingMultiProcessingDataLoaderIter
from jitlearn.tablet import RealtimeTabletDataset, UniformTabletDataset, \
    dataloader_from_dataset


def train(model, opt, train_pairs, eval_pairs):
    fig, axs = plt.subplots(1, 2)
    movie, target, _ = next(eval_pairs)
    _ = next(eval_pairs)
    out = movie.detach().cpu().numpy()
    middle_frame = (movie.shape[1] - 1) // 2
    obj0 = axs[0].imshow(
        out[0, middle_frame],
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')
    obj1 = axs[1].imshow(
        out[0, -1],
        interpolation=None,
        vmin=0.0, vmax=0.75,
        aspect='equal')

    def update(i):
        model.train()
        for j in range(1000):
            print('.', end='')
            movie, target, info = next(train_pairs)
            opt.zero_grad()
            out = model(movie.cuda())
            loss = model.crit(out, target.cuda())
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            eval_item = next(eval_pairs)
            movie, target, info = eval_item
            out = model(movie.cuda())
            out = out.detach().cpu().numpy()

        obj0.set_array(target[0, -1])
        obj1.set_array(out[0, -1])
        return [obj0, obj1]

    anim = FuncAnimation(fig, update)
    plt.show()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    START_TIME = time() + 15
    BUFFER_LEN = 32  # TODO

    train_ds = RealtimeTabletDataset(
        path,
        settings_path,
        range(PROJ_IDS.start, PROJ_IDS.start + 10000), #JUMP_AT),
        # PROJ_IDS,
        TRAIN_VOXELS,
        POS_MIN_BOTTOM,
        POS_MAX_BOTTOM,
        ROT_MIN,
        ROT_MAX,
        time_per_proj=TIME_PER_PROJ,
        start_time=START_TIME,  # start train and eval at same time
        sequence_length=SEQ_LEN,
        sequence_stride=SEQ_STRIDE,
        device_ids=[1],
        iso_voxel_size=ISO_VOXEL_SIZE,
        preload=True)
    train_dl = dataloader_from_dataset(train_ds, batch_size=BATCH_SIZE)

    visu_ds = RealtimeTabletDataset(
        path,
        settings_path,
        PROJ_IDS,
        (250, 250, VOXELS_Z),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        time_per_proj=TIME_PER_PROJ,
        start_time=START_TIME,  # start train and eval at same time
        sequence_length=SEQ_LEN,
        sequence_stride=SEQ_STRIDE,
        device_ids=[2],
        iso_voxel_size=ISO_VOXEL_SIZE,
        preload=False,
    )
    visu_dl = dataloader_from_dataset(visu_ds, batch_size=1)

    # These datasets start when __iter__ is called
    train_pairs = iter(BufferingMultiProcessingDataLoaderIter(
        train_dl, max_len=BUFFER_LEN))
    visu_pairs = iter(BufferingMultiProcessingDataLoaderIter(visu_dl, 1))
    # train_pairs = iter(train_dl)
    # visu_pairs = iter(visu_dl)

    train(model.cuda(),
          optim.Adam(model.parameters(), lr=LR),
          train_pairs, visu_pairs)
