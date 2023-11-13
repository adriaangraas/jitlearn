import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({'figure.raise_window': False})
plt.style.use('dark_background')

import torch

from spatiotemporal.rayveds import BufferingMultiProcessingDataLoaderIter


def realtime(model, params, opt, pairs, buffer_size, pairs_eval):
    # They *must* be started at the same time!
    pairs_iter = BufferingMultiProcessingDataLoaderIter(pairs, buffer_size)
    pairs_train = enumerate(pairs_iter)

    try:
        pairs_eval = BufferingMultiProcessingDataLoaderIter(pairs_eval)
    except:
        pairs_eval = iter(pairs_eval)

    fig = plt.figure(2)
    objs = []
    objs = None

    def update(i, objs=objs):
        for j in range(10):
            train_item = next(pairs_train)
            _, (movie, target, signal) = train_item

            model.train()
            opt.zero_grad()
            out = model(movie.cuda(), {k: v.cuda() for k, v in signal.items()})
            loss = model.crit(out, target.cuda())
            loss.backward()
            opt.step()
            model.cpu()
            params.update(model.state_dict())
            model.cuda()

        eval_item = next(pairs_eval)
        movie, target, signal = eval_item
        model.eval()
        out = model(movie.cuda(), {k: v.cuda() for k, v in signal.items()})
        out = torch.squeeze(out).detach().cpu().numpy()

        if objs is None:
            objs = plt.gca().imshow(
                out,
                interpolation=None,
                vmin=0.0, vmax=0.75,
                aspect='equal')
        else:
            objs.set_data(out)

        # assert len(movie) == 32
        # if len(objs) == 0:
        #     for ax, im in zip(grid, movie[:16]):
        #         objs.append(ax.imshow(im[0, ...], interpolation=None,
        #                               vmin=0.0, vmax=0.75))
        #     for ax, obj in zip(grid, objs):
        #         ax.draw_artist(obj)
        # else:
        #     for ax, obj, im in zip(grid, objs, movie[:16]):
        #         obj.set_data(im[0, ...])
        #         ax.draw_artist(obj)

        return [objs]

    FuncAnimation(fig, update, None, interval=5, blit=True)
    plt.show()


if __name__ == '__main__':
    # Note: include the examples directory in the PYTHONPATH
    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)
    from dae import RealtimeTopToBottomTest2

    basedir = "/export/scratch2/adriaan/zond_fastdvdnet_training"
    orientation = torch.zeros((9,))
    orientation.share_memory_()
    expt = RealtimeTopToBottomTest2(orientation)
    model = expt.model
    state = torch.multiprocessing.Manager().dict(model.state_dict())
    model.cuda()
    realtime(
        model,
        state,
        expt.opt,
        expt.pairs_train,
        expt.buffer_size,
        expt.pairs_eval)
