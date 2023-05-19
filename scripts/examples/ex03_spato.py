from time import sleep, time

import matplotlib.pyplot as plt
import torch
from matplotlib import animation

from jitlearn import BufferingMultiProcessingDataLoaderIter
from jitlearn.tablet import UniformTabletDataset, dataloader_from_dataset, \
    RealtimeTabletDataset

if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    MAIN_DIR = 'handsoap'
    SUB_DIR = 'scan_1'
    TEMPLATE_PATH = "/export/scratch3/adriaan/bigstore/{main_dir}/{sub_dir}"
    path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir=SUB_DIR)
    settings_path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir='pre_scan')

    SEQ_LEN = 3
    SEQ_STRIDE = 150

    ds = UniformTabletDataset(
        path,
        settings_path,
        range(1000, 4000),
        (20, 20, 1),  # note: 4D is supported but untested, keep z-dim 1
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        # time_per_proj=0.012,
        # start_time=time() + 10,
        sequence_length=SEQ_LEN,
        sequence_stride=SEQ_STRIDE,
        device_ids=[0],
        iso_voxel_size=0.25,
        preload=True)
    dl = dataloader_from_dataset(ds, batch_size=1)

    # note: for a realtime dataset, use our custom iterator to continuously
    # generate samples, and specify the size of the buffer
    # pairs = iter(BufferingMultiProcessingDataLoaderIter(dl, 10))
    pairs = iter(ds)


    def next_item():
        train_item = next(pairs)
        input, target, info = train_item
        # input = torch.squeeze(input).detach().numpy()
        return input, info

    from tqdm import tqdm
    import itertools
    for _ in tqdm(itertools.count()):
        input, info = next_item()

    fig, axs = plt.subplots(1, SEQ_LEN)
    objs = []
    for t in range(SEQ_LEN):
        axs[t].set_title(f"Time {t + 1}")
        objs.append(axs[t].imshow(
            input[t],
            interpolation=None,
            vmin=0.0, vmax=0.75,
            aspect='equal'))

    def update(i):
        input, info = next_item()
        print(f"Current frame: {info['num_start'].numpy()[0]}")
        for t, obj in enumerate(objs):
            obj.set_array(input[t])
        return objs

    _ = animation.FuncAnimation(fig, update, blit=True)
    plt.show()