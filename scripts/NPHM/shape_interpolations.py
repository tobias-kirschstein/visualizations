from math import ceil
from pathlib import Path

import numpy as np
import pyvista as pv

import trimesh
from elias.util.io import save_img
from tqdm import tqdm
from trimesh import transformations

from visualizations.env_nphm import NPHM_DATA_PATH
from visualizations.math.vector import Vec3
from visualizations.render.single_mesh import render_single_mesh

NPHM_SHAPE_INTERPOLATIONS_FOLDER = "//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/interpolations"
# Contains mesh_0000.ply -> mesh_1847.ply
# with shape latent code interpolations

SELECTED_INTERPOLATIONS = ["shape_interpol_val_part1_moreSteps"]

NPHM_SHAPE_INTERPOLATIONS_RENDERINGS_FOLDER = f"{NPHM_DATA_PATH}/shape_interpolations"

USE_BLENDER = True
RESOLUTION = 1024
IMAGE_WIDTH = RESOLUTION
IMAGE_HEIGHT = RESOLUTION
N_ROTATIONS = 1
ROTATION_START = 1  # percentage of video length until rotations start. If >1, the video will be repeated once, and the rotation starts in the second loop
SKIP_EVERY_NTH_FRAME = 21  # During marching cubes, the corner points of the interpolation were generated twice

"""
To create video:
ffmpeg -framerate 50 -i .\shape_interpolation_%05d.png -pix_fmt yuv420p out.mp4
"""

def _create_shape_interpolation(interpolation_name: str):
    interpolation_folder = f"{NPHM_SHAPE_INTERPOLATIONS_FOLDER}/{interpolation_name}"
    assert Path(interpolation_folder).exists(), \
        f"Could not find {interpolation_folder}. Check the path and whether it is accessible"
    mesh_paths = list(sorted(Path(interpolation_folder).glob('mesh_*.ply')))

    n_frames = len(mesh_paths)
    if SKIP_EVERY_NTH_FRAME > 0:
        n_groups = ceil(n_frames / SKIP_EVERY_NTH_FRAME)
        n_frames = n_frames - n_groups + 1  # Skip 1 frame per group except for the first group

    n_repeats = 1
    if ROTATION_START >= 1:
        n_repeats += 1

    rotation_start = int(n_frames * ROTATION_START)
    frame_id = 0
    for i_repeat in range(n_repeats):
        for i_mesh, mesh_path in tqdm(enumerate(mesh_paths)):
            if SKIP_EVERY_NTH_FRAME > 0 and i_mesh % SKIP_EVERY_NTH_FRAME == 0 and i_mesh > 0:
                # Always keep the very first frame, but then skip every n-th frame as it is a duplicate
                # Basically, we skip the first frame of a group of n frames (0, 21, 42, 63, ...) but don't do it for
                # the very first frame as the previous frame does not exist (would be -1) and hence it actually is not
                # a duplicate
                continue

            angle = 0
            if frame_id >= rotation_start and ROTATION_START > 0:
                n_rotation_frames = n_repeats * n_frames - rotation_start
                n_frames_per_rotation = n_rotation_frames / N_ROTATIONS
                rotation_per_frame = 2 * np.pi / n_frames_per_rotation
                angle = rotation_per_frame * (frame_id - rotation_start)

            rendered_mesh_path = f"{NPHM_SHAPE_INTERPOLATIONS_RENDERINGS_FOLDER}/{interpolation_name}/shape_interpolation_{frame_id:05d}.png"
            if not Path(rendered_mesh_path).exists():
                if USE_BLENDER:

                    rendered_mesh = render_single_mesh(mesh_path,
                                                       image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                                       use_orthographic_cam=False,
                                                       angle=angle,
                                                       camera_distance=2,
                                                       mirror_light_x=True,
                                                       mirror_light_z=True,
                                                       location=Vec3(0, 0, 0.2))

                else:
                    p = pv.Plotter(off_screen=True)
                    mesh = trimesh.load(mesh_path)
                    rotation = transformations.rotation_matrix(angle, [0, 1, 0])
                    mesh.apply_transform(rotation)
                    p.camera_set = True
                    p.camera_position = 'xy'
                    p.camera.position = (0, 0, 3)
                    p.add_mesh(mesh)
                    rendered_mesh = p.screenshot(transparent_background=True)
                    del mesh
                    p.clear()
                    p.deep_clean()

                save_img(rendered_mesh, rendered_mesh_path)
                del rendered_mesh

            frame_id += 1


if __name__ == '__main__':

    for selected_interpolation in SELECTED_INTERPOLATIONS:
        _create_shape_interpolation(selected_interpolation)
