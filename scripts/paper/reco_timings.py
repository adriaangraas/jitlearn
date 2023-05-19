import time
from matplotlib.ticker import StrMethodFormatter

from jitlearn import UniformTabletDataset, dataloader_from_dataset
from jitlearn.plotting import CM, plt
import numpy as np
from tqdm import tqdm


def timing(ds, warmup=200, iters=200):
    for _ in tqdm(range(warmup)):
        next(ds)

    samples = []
    for _ in tqdm(range(iters)):
        start = time.time()
        i, sample = next(ds)
        elapsed = time.time() - start
        samples.append(elapsed)

        # 100: 36.3 it/s
        # 200, 200, 1: 36.8
        # 200, 200, 10: 35.60
        # 400, 400, 1: 35.5
        # 400, 400, 10: 29.60
        # 20, 20, 1: 37.4?
        # 20, 20, 10: 37.0
        # 60, 60, 1: 37.4?
        # 60, 60, 10: 37.0
        # 1000, 1000, 10: 10 it/s
        # 1024, 1024, 1024: 10 it/s
        # 2000, 2000, 1: 17.0 it/s
        # 2000, 2000, 10: 3.4 it/s

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(im[0][0, ..., 0], vmin=0, vmax=0.025 * 300 / V)
        # plt.show()

    mean = np.array(samples).mean()
    std = np.std(np.array(samples))
    return mean, std


def run():
    prefix = "/export/scratch3/adriaan/bigstore"
    # path = f"{prefix}/handsoap_layeredSand/scan_1"
    # settings_path = f"{prefix}/handsoap_layeredSand/pre_scan"
    path = f"{prefix}/massagegel/scan_4"
    settings_path = f"{prefix}/massagegel/pre_scan"

    def _ds(voxels):
        ds = UniformTabletDataset(
            path,
            settings_path,
            range(1000, 2000),
            voxels,
            (0., 0., 0.),
            (0., 0., 0.),
            (0., 0., 0.),
            (0., 0., 0.),
            device_ids=[0],
            iso_voxel_size=300 / voxels[0] * 0.25,
            preload=True
        )
        pairs_train = enumerate(ds)
        next(pairs_train)  # first item is slow

        # ds = rayveds(path,
        #              settings_path,
        #              range(16000, 17000),
        #              reco_interval=150,
        #              iso_voxel_size=300 / V * 0.25,
        #              device_ids=[0],
        #              voxels=voxels,
        #              preload=True,
        #              min_load=1000,
        #              sampler=RandomSampler(
        #                  (0., 0., 0.),
        #                  (0., 0., 0.),
        #                  (0., 0., 0.),
        #                  (0., 0., 0.),
        #              )
        # )
        # ds = enumerate(ds)
        # next(ds)  # first item is slow
        return pairs_train


    # slices = []
    # for s in range(100, 2100, 100):
    #     ds = _ds((s, s, 1))
    #     slices.append(timing(ds))
    # np.save('timing_slices.npy', slices)

    # slabs = []
    # for s in range(1, 26):
    #     ds = _ds((1000, 1000, s))
    #     slabs.append(timing(ds))
    # np.save('timing_slabs.npy', slabs)

    volumes = []
    for s in range(100, 1100, 100):
        ds = _ds((s, s, s))
        volumes.append(timing(ds, warmup=1, iters=10))
    np.save('timing_volumes.npy', volumes)

    patches = []
    for s in range(20, 110, 10):
        ds = _ds((s, s, 1))
        patches.append(timing(ds))
    np.save('timing_patches.npy', patches)


    patchslabs = []
    for s in range(1, 26):
        ds = _ds((50, 50, s))
        patchslabs.append(timing(ds))
    np.save('timing_patchslab.npy', patchslabs)


if __name__ == '__main__':
    run()

    kwargs = {'markersize': 2,
              'fmt': '.',
              'color': 'k',
              'linewidth': .5
              }

    def _ticklabels(ax=None):
        if ax is None:
            ax = plt.gca()

        # ax.yaxis.set_major_formatter(
        #     StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
        # ax.yaxis.set_minor_formatter(
        #     StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 0))

    s = "x"
    def convert(data):
        # for i, d in enumerate(data):
        #     data[i][0] = 1 / d[0]
        #     data[i][1] = 1 / d[1]
        return data * 1000  # to ms

    def convert_dnn(data):
        return data

    fig, axs = plt.subplots(
        nrows=2, ncols=4,
        figsize=(18 * CM, 6 * CM)
    )
    slices = convert(np.load('timing_slices.npy'))
    axs[1, 0].set_title(f"${s}, {s}$")
    axs[1, 0].errorbar(
        range(100, 2100, 100),
        [s[0] for s in slices],
        yerr=[s[1] for s in slices],
        **kwargs
    )
    _ticklabels(axs[0,0])

    slabs = convert(np.load('timing_slabs.npy'))
    axs[1, 1].set_title(f"$1000, 1000, {s}$")
    axs[1, 1].errorbar(
        range(1, 26),
        [s[0] for s in slabs],
        yerr=[s[1] for s in slabs],
        **kwargs
    )
    _ticklabels(axs[0,1])
    patches = convert(np.load('timing_patches.npy'))
    # axs[1, 0].yaxis.set_major_locator(plt.MultipleLocator(1.))
    axs[0, 0].set_title(f"${s}, {s}$")
    axs[0, 0].errorbar(
        range(20, 110, 10),
        [s[0] for s in patches],
        yerr=[s[1] for s in patches],
        **kwargs
    )
    _ticklabels(axs[1, 0])

    patchslabs = convert(np.load('timing_patchslab.npy'))
    axs[0, 1].set_title(f"$50, 50, {s}$")
    axs[0, 1].errorbar(
        range(1, 26),
        [s[0] for s in patchslabs],
        yerr=[s[1] for s in patchslabs],
        **kwargs
    )
    _ticklabels(axs[1, 1])

    kwargs |= {'fmt': 'x'}

    slices = convert_dnn(np.load('dnntiming_slices.npy'))
    axs[1, 2].set_title(f"${s}, {s}$")
    axs[1, 2].errorbar(
        range(1000, 2100, 100),
        [s[0] for s in slices],
        yerr=[s[1] for s in slices],
        **kwargs)
    _ticklabels(axs[0,0])

    slabs = convert_dnn(np.load('dnntiming_slabs.npy'))
    axs[1, 3].set_title(f"$1000, 1000, {s}$")
    axs[1, 3].errorbar(
        range(1, 17, 2),
        [s[0] for s in slabs],
        yerr=[s[1] for s in slabs],
        **kwargs)
    _ticklabels(axs[0,1])

    axs[0, 2].set_title(f"${s}, {s}$")
    # patches = np.load('dnntiming_patches.npy')
    # axs[1, 2].errorbar(
    #     range(20, 110, 10),
    #     [s[0] for s in patches],
    #     yerr=[s[1] for s in patches],
    #     **kwargs)
    patches_train = convert_dnn(np.load('dnntiming_patches_train.npy'))
    axs[0, 2].errorbar(
        range(20, 110, 10),
        [s[0] for s in patches_train],
        yerr=[s[1] for s in patches_train],
        **(kwargs | {'fmt': '^'}))
    _ticklabels(axs[1,0])

    axs[0, 3].set_title(f"$50, 50, {s}$")
    # patchslabs = np.load('dnntiming_patchslab.npy')
    # axs[1, 3].errorbar(
    #     range(1, 26, 2),
    #     [s[0] for s in patchslabs],
    #     yerr=[s[1] for s in patchslabs],
    #     **kwargs)
    patchslabs_train = convert_dnn(np.load('dnntiming_patchslab_train.npy'))
    axs[0, 3].errorbar(
        range(1, 26, 2),
        [s[0] for s in patchslabs_train],
        yerr=[s[1] for s in patchslabs_train],
        **(kwargs | {'fmt': '^'}))
    _ticklabels(axs[1,1])

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.915,
        bottom=0.075,
        left=0.053,
        right=0.995,
        hspace=0.6,
        wspace=0.3
    )
    plt.savefig('timings.pdf')
    plt.show()

    # plt.tight_layout()
    # plt.savefig('recotimings.pdf')
    # plt.show()
