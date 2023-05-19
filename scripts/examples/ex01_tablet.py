import matplotlib.pyplot as plt
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

    # ds = ContinuousTabletDataset(
    #     path,
    #     settings_path,
    #     range(1000, 2000),
    #     (200, 200, 1),
    #     (0., 0., 0.),
    #     (0., 0., 0.),
    #     (0., 0., 0.),
    #     (0., 0., 0.),
    #     device_ids=[1],
    #     iso_voxel_size=0.25,
    #     num_increment=15,
    # )

    ds = UniformTabletDataset(
        path,
        settings_path,
        range(1000, 13000),
        (200, 200, 3),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0., 0.),
        device_ids=[1],
        iso_voxel_size=0.25,
    )

    # note: for a continuous dataset, select batch_size=1 to have a
    # time increment of `num_increment` per batch.
    pairs_train = dataloader_from_dataset(ds, batch_size=1)
    pairs_train = iter(pairs_train)


    def next_item():
        train_item = next(pairs_train)
        input, target, info = train_item
        input = torch.squeeze(input).detach().numpy()
        target = torch.squeeze(target).detach().numpy()
        print(f"Current frame: {info['num_start'].numpy()[0]}")
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
        obj0.set_array(input)
        obj1.set_array(target)
        return obj0, obj1

    _ = animation.FuncAnimation(fig, update, blit=True)
    plt.show()
