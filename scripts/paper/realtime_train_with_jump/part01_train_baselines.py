import itertools
from functools import partial

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import optim
import cupy as cp
from tqdm import tqdm

from config import *
from jitlearn import BufferingMultiProcessingDataLoaderIter

from jitlearn.tablet import TabletDataset, UniformTabletDataset, \
    _worker_init_fn, dataloader_from_dataset
from jitlearn.training import eval_dataloader, save


def train(model, opt, train_pairs, eval_pairs_list, fname, batches):
    noisy, denoised = [], []

    # fig = plt.figure()
    # fig2, axs = plt.subplots(1, 2)

    batch = 0
    for _ in itertools.count():
        model.train()
        for _ in range(100):
            batch += 1
            print('.', end='')
            movie, target, info = next(train_pairs)
            opt.zero_grad()
            out = model(movie.cuda())
            loss = model.crit(out, target.cuda())
            loss.backward()
            opt.step()

        model.eval()
        # movie, target, info = next(eval_pairs)
        # obj0.set_array(target.detach().cpu().numpy()[0, -1])
        # obj1.set_array(model(movie.cuda()).detach().cpu().numpy()[0, 0])
        noisy_agg, denoised_agg = eval_dataloader(
            model, iter(eval_pairs_list), len(eval_pairs_list))

        noisy += [noisy_agg.average()]
        denoised += [denoised_agg.average()]
        # plt.cla()
        # plt.plot(noisy)
        # plt.plot(denoised)
        # plt.pause(.001)

        # model.eval()
        # with torch.no_grad():
        #     eval_item = next(eval_pairs)
        #     movie, target, info = eval_item
        #     out = model(movie.cuda()).detach().cpu().numpy()
        #     plt.cla()
        #     axs[0].imshow(
        #         target[0, -1],
        #         interpolation=None,
        #         vmin=0.0, vmax=0.75,
        #         aspect='equal')
        #     axs[1].imshow(
        #         out[0, -1],
        #         interpolation=None,
        #         vmin=0.0, vmax=0.75,
        #         aspect='equal')
        #     plt.pause(.01)

        save(fname,
             model,
             opt,
             batch,
             allow_overwrite=True,
             noisy=noisy_agg.average(),
             denoised=denoised_agg.average())

        if batch == batches:
            break


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    # Train two baselines
    #  - one from [start, jump]
    #  - other on [jump, end]
    # while making sure we have enough BATCHES to find an optimum.

    BATCHES = 40000
    EVAL_BATCHES = 1000  # * 32 = 32000 samples
    exp1 = (range(PROJ_IDS.start, JUMP_AT),
            range(JUMP_AT + 1000, JUMP_AT + 2000),
            "./baseline_before_jump_r2_batch{}.ckpt",
            POS_MIN_BOTTOM,
            POS_MAX_BOTTOM,
            )
    exp2 = (range(JUMP_AT, PROJ_IDS.stop),
            range(JUMP_AT - 2000, JUMP_AT - 1000),
            "./baseline_after_jump_r2_batch{}.ckpt",
            POS_MIN_TOP,
            POS_MAX_TOP,
            )

    for exp in (exp1, exp2):
        # training dataset
        train_ds = UniformTabletDataset(
            path,
            settings_path,
            exp[0],
            TRAIN_VOXELS,
            exp[3],
            exp[4],
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

        # for eval dataset we take 1000 projections after the jump,
        # 1000 projections away
        eval_ds = UniformTabletDataset(
            path,
            settings_path,
            exp[1],
            TRAIN_VOXELS,
            exp[3],
            exp[4],
            ROT_MIN,
            ROT_MAX,
            sequence_length=SEQ_LEN,
            sequence_stride=SEQ_STRIDE,
            device_ids=[2],
            iso_voxel_size=ISO_VOXEL_SIZE,
            preload=True
        )
        eval_pairs = dataloader_from_dataset(eval_ds, batch_size=BATCH_SIZE)

        eval_pairs = iter(eval_pairs)
        eval_pairs_list = [i for _, i in tqdm(zip(range(EVAL_BATCHES), eval_pairs),
                    desc="Generating evaluation dataset...")]
        train(model.cuda(), optim.Adam(model.parameters(), lr=LR), train_pairs,
              eval_pairs_list, exp[2], BATCHES)