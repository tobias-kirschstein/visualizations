from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pyvista as pv
import trimesh
from elias.util.io import save_img
from matplotlib import pyplot as plt
from scipy.spatial.transform.rotation import Rotation as R
from tqdm import tqdm
from trimesh import transformations

from visualizations.env_nphm import NPHM_DATA_PATH
from visualizations.render.single_mesh import render_single_mesh

NPHM_RESULTS_IDENTITY_FOLDER = f"{NPHM_DATA_PATH}/results_identity"
NPHM_RESULTS_IDENTITY_ABLATION_FOLDER = f"{NPHM_DATA_PATH}/results_identity_ablation"
NPHM_RESULTS_IDENTITY_ABLATION_VIDEO_RENDERINGS_FOLDER = f"{NPHM_RESULTS_IDENTITY_ABLATION_FOLDER}/video"

USE_BLENDER = True

RESOLUTION = 1024
IMAGE_WIDTH = int(3 / 4 * RESOLUTION)
IMAGE_HEIGHT = RESOLUTION

N_FRAMES = 125
ANGLE = np.pi / 2

METHODS = ["nP", "npm", 'PC', 'GT']
SELECTED_ABLATIONS = ["fixed_npoints500"]
SELECTED_IDENTITIES = ["id95neutral_1_highRe"]


def _lin_interpolate(alpha: float,
                     start: float, end: float,
                     alpha_start: float = 0, alpha_end: float = 1) -> float:
    alpha_len = alpha_end - alpha_start
    return (alpha - alpha_start) / (alpha_len) * (end - start) + start


def _angle_fn(alpha: float):
    step_1 = 0.25
    step_2 = 0.75

    if alpha < step_1:
        # interpolate from 0 -> ANGLE / 2
        return _lin_interpolate(alpha, 0, ANGLE / 2, alpha_end=step_1)
    elif alpha < step_2:
        # Interpolate from ANGLE / 2 -> -ANGLE / 2
        return _lin_interpolate(alpha, ANGLE / 2, -ANGLE / 2, alpha_start=step_1, alpha_end=step_2)
    else:
        # Interpolate from -ANGLE / 2 -> 0
        return _lin_interpolate(alpha, -ANGLE / 2, 0, alpha_start=step_2)


def _scaled_cmap(x: float, cmap, scale: float, brightness: float = 1) -> np.ndarray:
    cmap_result = cmap(x * scale)
    cmap_result[:, :3] *= brightness
    cmap_result = cmap_result.clip(0, 1)
    return cmap_result


def scaled_cmap(cmap_name: str, scale: float, brightness: float = 1):
    cmap = plt.get_cmap(cmap_name)
    return lambda x: _scaled_cmap(x, cmap=cmap, scale=scale, brightness=brightness)


def load_mesh(ablation: str, method: str, identity: str) -> Tuple[str, Union[np.ndarray, trimesh.Trimesh]]:
    mesh = None
    ablation_parts = ablation.split('_')
    if len(ablation_parts) == 2:
        # only npoints, e.g., fixed_npoints500
        n_points = ablation_parts[1][len("npoints"):]
        noise = "None"
    else:
        # npoints and noise, e.g., fixednoise0.00025_npoints5000
        noise = ablation_parts[1][len("fixednoise"):]
        n_points = ablation_parts[2][len("npoints"):]

    # Identity is long, e.g., id95neutral_1_highRe
    # We only want the "id95" part
    identity_parts = identity.split('_')
    identity_short = identity_parts[0][:-len("neutral")]
    expression_id = identity_parts[1]

    if method == 'PC':
        mesh_path = f"{NPHM_RESULTS_IDENTITY_ABLATION_FOLDER}/pointclouds2/{identity_short}_{expression_id}_noise{noise}_numPoints_{n_points}.npy"
        mesh = np.load(mesh_path)
        # pointclouds in .npy files are already rotated by 50°. Invert this rotation here, so the pointclouds
        # align with the other meshes
        rotation_matrix_1 = R.from_euler(seq='xyz', angles=[0, -50 / 360 * 2 * np.pi, 0]).as_matrix()
        # mesh = mesh @ rotation_matrix_1.T
    else:
        if method == 'GT':
            mesh_path = f"{NPHM_RESULTS_IDENTITY_FOLDER}/GT/mesh_s_{identity_short}_e_{expression_id}.ply"
        else:
            mesh_path = f"{NPHM_RESULTS_IDENTITY_ABLATION_FOLDER}/{method}_{ablation}/{identity}.ply"

        if not USE_BLENDER:
            mesh = trimesh.load(mesh_path)

    return mesh_path, mesh


def _generate_frames_for_method(ablation: str, method: str, identity: str):

    mesh_path, mesh = load_mesh(ablation, method, identity)

    for frame_id in tqdm(range(N_FRAMES)):

        angle = _angle_fn(frame_id / N_FRAMES)
        rendered_mesh_path = f"{NPHM_RESULTS_IDENTITY_ABLATION_VIDEO_RENDERINGS_FOLDER}/{ablation}/{identity}/{method}/frame_{frame_id:05d}.png"

        if not Path(rendered_mesh_path).exists():
            if USE_BLENDER and not method == 'PC':
                rendered_mesh = render_single_mesh(mesh_path,
                                                   image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                                   use_orthographic_cam=False,
                                                   angle=angle,
                                                   camera_distance=2,
                                                   mirror_light_x=True)
            else:

                p = pv.Plotter(off_screen=True)

                p.camera_set = True
                p.camera_position = 'xy'
                p.camera.position = (0, 0, 3)
                p.background_color = (0, 0, 0)
                # Note: taking negative angle here, as blender version rotates the camera, while pyvista rotates the obj
                rotation = transformations.rotation_matrix(-angle, [0, 1, 0])
                if method == 'PC':
                    depths = mesh[:, 2]
                    points = mesh @ rotation[:3, :3].T
                    p.theme.render_points_as_spheres = True
                    p.add_points(points,
                                 scalars=-depths,
                                 point_size=5,
                                 cmap=scaled_cmap('turbo', 1, brightness=1.3),
                                 clim=[-0.6, 0.6])
                    p.remove_scalar_bar()
                    rendered_mesh = p.screenshot(transparent_background=True)
                else:
                    mesh_copy = mesh.copy()
                    mesh_copy.apply_transform(rotation)
                    p.add_mesh(mesh_copy)
                    rendered_mesh = p.screenshot(transparent_background=True)
                    del mesh_copy

                p.clear()
                p.deep_clean()
                pv.close_all()

            save_img(rendered_mesh, rendered_mesh_path)


if __name__ == '__main__':

    for ablation in SELECTED_ABLATIONS:
        for identity in SELECTED_IDENTITIES:
            for method in METHODS:
                _generate_frames_for_method(ablation, method, identity)
