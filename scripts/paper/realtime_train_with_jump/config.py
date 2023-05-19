import random
import torch
from unet import Unet3d
import numpy as np

TEMPLATE_PATH = "/export/scratch3/adriaan/bigstore/{main_dir}/{sub_dir}"
MAIN_DIR = 'massagegel'
SUB_DIR = 'scan_3'
PROJ_IDS = range(1000, 24000)
JUMP_AT = 15000
SEQ_LEN = 3
SEQ_STRIDE = 60
TRAIN_VOXELS = (20, 20, 1)
ISO_VOXEL_SIZE = .25
VOXELS_X = TRAIN_VOXELS[0]
VOXELS_Z = TRAIN_VOXELS[-1]
ROT_MIN = (0., 0., np.pi)
ROT_MAX = (0., 0., np.pi)
_padding_xy = ISO_VOXEL_SIZE * VOXELS_X / 2
_posmin = (-18 + _padding_xy, -18 + _padding_xy, -40 + _padding_xy)
_posmax = (18 - _padding_xy, 18 - _padding_xy, 40 - _padding_xy)
POS_MIN = [np.min((0, s)) for s in _posmin]
POS_MAX = [np.max((0, s)) for s in _posmax]
_posmin = (-18 + _padding_xy, -18 + _padding_xy, -40 + _padding_xy)
_posmax = (18 - _padding_xy, 18 - _padding_xy, 0 - _padding_xy)
POS_MIN_BOTTOM = [np.min((0, s)) for s in _posmin]
POS_MAX_BOTTOM = [np.max((0, s)) for s in _posmax]
_posmin = (-18 + _padding_xy, -18 + _padding_xy, 0 + _padding_xy)
_posmax = (18 - _padding_xy, 18 - _padding_xy, 40 - _padding_xy)
POS_MIN_TOP = [np.min((0, s)) for s in _posmin]
POS_MAX_TOP = [np.max((0, s)) for s in _posmax]
TIME_PER_PROJ = 0.012 * 8
LR = 1e-4
BATCH_SIZE = 32

path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir=SUB_DIR)
settings_path = TEMPLATE_PATH.format(main_dir=MAIN_DIR, sub_dir='pre_scan')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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