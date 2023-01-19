from pathlib import Path
from typing import Tuple, Union

import cairo
import numpy as np
import pyvista as pv
import trimesh
from cairo import Format
from elias.util.io import save_img, load_img
from matplotlib import pyplot as plt
from scipy.spatial.transform.rotation import Rotation as R

from visualizations.cairo.image import draw_image, to_image
from visualizations.env_nphm import NPHM_DATA_PATH
from visualizations.math.vector import Vec2
from visualizations.render.single_mesh import render_single_mesh

NPHM_RESULTS_IDENTITY_ABLATION_FOLDER = f"{NPHM_DATA_PATH}/results_identity_ablation"
NPHM_RESULTS_IDENTITY_FOLDER = f"{NPHM_DATA_PATH}/results_identity"
NPHM_RESULTS_IDENTITY_ABLATION_RENDERINGS_FOLDER = f"{NPHM_RESULTS_IDENTITY_ABLATION_FOLDER}/renderings"

USE_BLENDER = True
RESOLUTION = 1024
IMAGE_WIDTH = int(3 / 4 * RESOLUTION)
IMAGE_HEIGHT = RESOLUTION
FIGURE_WIDTH = 2048
OVERLAP_X = 0.3
OVERLAP_Y = 0.3

SELECTED_NOISE_ABLATIONS = ["fixednoise0.0025_npoints5000", "fixednoise0.00025_npoints5000", "fixed_npoints5000", "GT"]
SELECTED_NOISE_METHODS = ["PC", "npm", "nP"]
SELECTED_NOISE_IDENTITIES = ["id95neutral_1_highRe"]

SELECTED_N_POINTS_ABLATIONS = ["fixed_npoints500", "fixed_npoints1000", "fixed_npoints5000", "GT"]
SELECTED_N_POINTS_METHODS = ["PC", "npm", "nP"]
SELECTED_N_POINTS_IDENTITIES = ["assianeutral_8_highRe"]


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
    if not ablation == 'GT':
        ablation_parts = ablation.split('_')
        # npoints and noise, e.g., fixednoise0.00025_npoints5000
        noise = ablation_parts[0][len("fixednoise"):]
        n_points = ablation_parts[1][len("npoints"):]
        if noise == '':
            noise = 'None'

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
        if ablation == 'GT':
            mesh_path = f"{NPHM_RESULTS_IDENTITY_FOLDER}/GT/mesh_s_{identity_short}_e_{expression_id}.ply"
        else:
            mesh_path = f"{NPHM_RESULTS_IDENTITY_ABLATION_FOLDER}/{method}_{ablation}/{identity}.ply"

        if not USE_BLENDER:
            mesh = trimesh.load(mesh_path)

    return mesh_path, mesh


if __name__ == '__main__':

    for ablation_type in {"noise", "n_points"}:
        if ablation_type == 'noise':
            SELECTED_IDENTITIES = SELECTED_NOISE_IDENTITIES
            SELECTED_ABLATIONS = SELECTED_NOISE_ABLATIONS
            SELECTED_METHODS = SELECTED_NOISE_METHODS
        elif ablation_type == 'n_points':
            SELECTED_IDENTITIES = SELECTED_N_POINTS_IDENTITIES
            SELECTED_ABLATIONS = SELECTED_N_POINTS_ABLATIONS
            SELECTED_METHODS = SELECTED_N_POINTS_METHODS
        else:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

        for identity in SELECTED_IDENTITIES:

            n_cols = len(SELECTED_ABLATIONS)
            n_rows = len(SELECTED_METHODS)

            cell_width = FIGURE_WIDTH / n_cols
            aspect_ratio = IMAGE_HEIGHT / IMAGE_WIDTH
            cell_height = aspect_ratio * cell_width
            figure_height = int(n_rows * cell_height)

            surface = cairo.ImageSurface(Format.ARGB32, FIGURE_WIDTH, figure_height)
            ctx = cairo.Context(surface)

            for i_ablation, ablation in enumerate(SELECTED_ABLATIONS):
                for i_method, method in enumerate(SELECTED_METHODS):

                    if ablation == 'GT' and method == 'PC':
                        continue
                    rendered_mesh_path = f"{NPHM_RESULTS_IDENTITY_ABLATION_RENDERINGS_FOLDER}/{ablation_type}/{identity}/{method}_{ablation}.png"

                    if not Path(rendered_mesh_path).exists():
                        mesh_path, mesh = load_mesh(ablation, method, identity)

                        if USE_BLENDER and not method == 'PC':
                            rendered_mesh = render_single_mesh(mesh_path,
                                                               image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                                               use_orthographic_cam=False,
                                                               camera_distance=2,
                                                               mirror_light_x=True,
                                                               crop_y_min=-17 / 25)
                        else:
                            p = pv.Plotter(off_screen=True)

                            p.camera_set = True
                            p.camera_position = 'xy'
                            p.camera.position = (0, 0, 2)
                            p.background_color = (0, 0, 0)
                            if method == 'PC':
                                rotation_matrix = R.from_euler(seq='xyz',
                                                               angles=[0, 50 / 360 * 2 * np.pi, 0]).as_matrix()
                                depths = mesh[:, 2]
                                mesh = mesh @ rotation_matrix.T
                                p.theme.render_points_as_spheres = True
                                p.add_points(mesh,
                                             scalars=-depths,
                                             point_size=5,
                                             cmap=scaled_cmap('turbo', 1, brightness=1),
                                             clim=[-0.6, 0.6])
                                p.remove_scalar_bar()
                            else:
                                p.add_mesh(mesh)
                            rendered_mesh = p.screenshot(transparent_background=True)

                        save_img(rendered_mesh, rendered_mesh_path)
                    else:
                        rendered_mesh = load_img(rendered_mesh_path)

                    center_x = i_ablation * cell_width + cell_width / 2
                    width = cell_width * (1 + OVERLAP_X)
                    height = cell_height * (1 + OVERLAP_Y)
                    if ablation == 'GT':
                        if i_method == 1:
                            center_y = figure_height / 2
                            draw_image(ctx, rendered_mesh, Vec2(center_x, center_y), size=Vec2(width, height))
                    else:
                        center_y = i_method * cell_height + cell_height / 2
                        draw_image(ctx, rendered_mesh, Vec2(center_x, center_y), size=Vec2(width, height))

            ablation_figure = to_image(surface)
            save_img(ablation_figure,
                     f"{NPHM_RESULTS_IDENTITY_ABLATION_RENDERINGS_FOLDER}/{ablation_type}/ablation_figure_{identity}.png")

    # for identity in SELECTED_N_POINTS_IDENTITIES:
    #     for ablation in SELECTED_N_POINTS_ABLATIONS:
    #         for method in SELECTED_N_POINTS_METHODS:
    #             rendered_mesh_path = f"{NPHM_RESULTS_IDENTITY_ABLATION_RENDERINGS_FOLDER}/n_points/{identity}/{method}_{ablation}.png"
    #
    #             if not Path(rendered_mesh_path).exists():
    #                 mesh_path, mesh = load_mesh(ablation, method, identity)
    #
    #                 if USE_BLENDER:
    #                     pass
    #                 else:
    #                     p = pv.Plotter(off_screen=True)
    #
    #                     p.camera_set = True
    #                     p.camera_position = 'xy'
    #                     p.camera.position = (0, 0, 3)
    #                     p.background_color = (0, 0, 0)
    #                     p.add_mesh(mesh)
    #                     rendered_mesh = p.screenshot(transparent_background=True)
    #
    #                 save_img(rendered_mesh, rendered_mesh_path)
    #             else:
    #                 rendered_mesh = load_img(rendered_mesh_path)
