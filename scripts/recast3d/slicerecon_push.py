import timeit

import numpy as np
import argparse
import reflex
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Push a FleX-ray data set to Slicerecon')

parser.add_argument('path',
                    metavar='path',
                    help='path to the data')
parser.add_argument('--settings',
                    metavar='settings',
                    help='path containing settings file',
                    default=None)
parser.add_argument('--detectorsample',
                    type=int,
                    default=1,
                    help='subselecting detector pixels')
parser.add_argument('--sample',
                    type=int,
                    default=1,
                    help='number of projections to skip')
parser.add_argument('--host',
                    default="localhost",
                    help='the projection server host')
parser.add_argument('--port',
                    type=int,
                    default=5558,
                    help='the projection server port')
parser.add_argument('--skipgeometry',
                    action='store_true',
                    help='assume the geometry packet is already sent')
parser.add_argument('--range',
                    nargs=2,
                    type=int,
                    help='projection numbers range for ReFleX')

args = parser.parse_args()
det_sample = args.detectorsample
sample = args.sample
settings_path = args.path if args.settings is None else args.settings

# Read dark, flats, proj
darks = reflex.darks(settings_path)
assert len(darks) > 0
flats = reflex.flats(settings_path)
assert len(flats) > 0
# proj_ids = range(1000, 1500)
slowdown_rate = 2
nr_angles_per_rot = 75
step = 150 // nr_angles_per_rot  # TODO: check

proj_ids = range(args.range[0], args.range[1], step)
projs = reflex.projs(args.path, proj_ids)

# darks[...] = 1.
# flats[...] = 1.
# projs[...] = 100.

settings = reflex.Settings.from_path(settings_path)
motor_geom = reflex.motor_geometry(
    settings)

# nr_sets = 2
# assert nr_sets == 1 or nr_sets == 2
# subset = 1
# assert subset == 0 or subset == 1
reconstruction_rotation_intval = 2 * np.pi

# NOTE:
# angles = (np.array(proj_ids) * reconstruction_rotation_intval / nr_angles_per_rot)
angles = (np.array(range(nr_angles_per_rot))
          * reconstruction_rotation_intval / nr_angles_per_rot)

# reco_interval = int(
#     np.ceil(2 * np.pi * nr_angles_per_rot / reconstruction_rotation_intval))

geom = reflex.circular_geometry(
    settings,
    initial_geom=motor_geom,
    angles=angles)
vectors = reflex.dynamic_geom_to_astra_vectors(geom)
vectors = [i for v in vectors for i in v]
proj_count, rows, cols = len(projs), settings.det_binned_rows(), settings.det_binned_cols()
proj_count = nr_angles_per_rot

import tomop
while True:
    publ = tomop.publisher(args.host, args.port)

    if not args.skipgeometry:
        # voxels = [geom.detector().rows, geom.detector().cols, geom.detector().cols]
        # w = p.settings.SDD() * geom.detector().width / 2 / p.settings.SOD()
        # h = p.settings.SDD() * geom.detector().height / 2 / p.settings.SOD()
        VOXEL_SIZE = 0.25
        WIDTH = 350  # voxels, set same to Slicerecon, i.e., --slice-size 350
        HEIGHT = 350  # voxels, set same to Slicerecon, i.e., --slice-size 350
        w = WIDTH * VOXEL_SIZE
        h = HEIGHT * VOXEL_SIZE

        packet_vol_geom = tomop.geometry_specification_packet(
            0,
            [-w / 2, -w / 2, -h / 2],
            [w / 2, w / 2, h / 2])
        publ.send(packet_vol_geom)

        # Scan settings
        already_linear = False
        packet_scan_settings = tomop.scan_settings_packet(
            0,
            darks.shape[0],
            flats.shape[0],
            already_linear)
        publ.send(packet_scan_settings)

        # Geometry
        packet_geometry = tomop.cone_vec_geometry_packet(0, rows, cols, proj_count, vectors)
        publ.send(packet_geometry)


    # Send darks, ...
    for i in np.arange(0, darks.shape[0]):
        publ.send(
            tomop.projection_packet(
                0, i, [rows, cols], np.ascontiguousarray(darks[i, :, :].flatten())))

    # ... flats, ...
    for i in np.arange(0, flats.shape[0]):
        publ.send(
            tomop.projection_packet(
                1, i, [rows, cols], np.ascontiguousarray(flats[i, :, :].flatten())))

    # ... and projections.
    from time import sleep
    r = [np.ascontiguousarray(proj.flatten()) for proj in projs]
    input("Press Enter to continue...")
    time_start = timeit.default_timer()
    for i, r_i in tqdm(enumerate(r)):
        time_since_start = timeit.default_timer() - time_start
        supposed_time = 0.012 * i * step * slowdown_rate
        if supposed_time > time_since_start:
            sleep(supposed_time - time_since_start)
        print(f"Current experiment time: {timeit.default_timer() - time_start}")
        publ.send(
            tomop.projection_packet(2, proj_count + i, [rows, cols], r_i))

    input("Press Enter to restart...")