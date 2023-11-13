import time

import tomop
import numpy as np

global vol

def _get_reco(recodir, fname, scale_factor):
    fname = recodir + fname
    reco = np.load(fname, allow_pickle=True)
    # x = reco['volume']
    # c = x.shape[2] // 2
    # x = x[..., c - 450:c + 350]
    reco = np.flip(reco, axis=-1)
    # y = denoise_tv_chambolle(y, weight=.15)
    # y *= 256
    print(np.min(reco))
    print(np.max(reco))
    reco *= 256 / np.max(reco) * scale_factor
    # np.clip(reco, 0, 256, out=reco)
    reco = np.clip(reco, 0, 256)
    return reco


def callback(orientation, slice_id):
    print("\n" * 3)
    print("-" * 100)
    print(orientation)
    print(vol.shape)
    print(slice_id)

    slice1 = vol[:, :, 200]
    shape = list(slice1.shape)
    return shape, slice1.flatten()


if __name__ == '__main__':
    # run_network()

    fname = f'out_sirt_massagegel_scan3_3000.npy'
    # fname = f'out_sirt_handsoap_scan1_3000.npy'
    vol = _get_reco('../spatiotemporal/', fname, 3.0)
    vol = np.copy(np.swapaxes(vol, 1, 2)).astype(np.float32)

    #
    M = np.max(vol.shape)
    vol2 = np.zeros([M] * 3)
    o1 = (M - vol.shape[0]) // 2
    o2 = (M - vol.shape[1]) // 2
    o3 = (M - vol.shape[2]) // 2
    vol2[o1:o1+vol.shape[0], o2:o2+vol.shape[1], o3:o3+vol.shape[2]] = vol
    vol = vol2

    serv = tomop.server("Massagegel")

    # slice1 = np.array(
    #     [
    #         [0, 255, 0, 255],
    #         [255, 0, 255, 0],
    #         [0, 0, 255, 255],
    #         [0, 0, 255, 255]], dtype=np.float32)

    # slice1 = vol[:, 100, :]

    # print('sending')
    # data = list(vol.astype(np.float32).flatten().tolist())
    # print('sending 2')

    # vdp = tomop.volume_data_packet(
    #     serv.scene_id(),
    #     np.array(np.flip(vol.shape), dtype='int32').tolist(),
    #     # np.array([0, 255, 128,
    #     #           255, 255,
    #     #           128, 255, 0], dtype='float32')
    #    data
    # )
    # print('sending 3')
    # serv.send(vdp)

    for i in range(100):
        slices = [vol[:, :, 200+i], vol[:, i, :], vol[200+i, :, :]]
        for i, slice in enumerate(slices):
            data = list(slice.astype(np.float32).flatten().tolist())
            sdp = tomop.slice_data_packet(
                serv.scene_id(),
                i, # slice id, I guess
                np.array(np.flip(slice.shape), dtype='int32').tolist(),
                data,
                False # additive
            )
            serv.send(sdp)
        time.sleep(.1)

    # print('setting callback')
    # serv.set_callback(callback)
    serv.serve()
