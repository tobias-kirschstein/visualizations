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

NPHM_ROTATE_HEAD_PATH = f"{NPHM_DATA_PATH}/rotate_head"

RESOLUTION = 1024
IMAGE_WIDTH = RESOLUTION
IMAGE_HEIGHT = RESOLUTION

USE_BLENDER = True
N_FRAMES = 150

MESH_PATHS = [
    ("barbara_hair",
     "//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/interpolations/shape_interpol_val_part1_moreSteps/mesh_0000.ply"),
    ("assia_smile",
     "//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/expression_interpol_test/expression_interpol_sorted_assia/mesh_0165.ply"),
    ("ziya_mouth_left",
     "//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/expression_transfer2/expression_interpol_sorted_ziya/mesh_0096.ply")
]

if __name__ == '__main__':

    for mesh_name, mesh_path in MESH_PATHS:
        mesh = trimesh.load(mesh_path)

        for frame_id in tqdm(range(N_FRAMES)):

            rendered_mesh_path = f"{NPHM_ROTATE_HEAD_PATH}/{mesh_name}/frame_{frame_id:05d}.png"

            if not Path(rendered_mesh_path).exists():

                angle = frame_id / N_FRAMES * 2 * np.pi
                if USE_BLENDER:
                    rendered_mesh = render_single_mesh(mesh_path,
                                                       image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                                       angle=angle, use_orthographic_cam=False,
                                                       camera_distance=2,
                                                       mirror_light_x=True,
                                                       mirror_light_z=True,
                                                       location=Vec3(0, 0, 0.2))
                else:

                    p = pv.Plotter(off_screen=True)
                    p.camera_set = True
                    p.camera_position = 'xy'
                    p.camera.position = (0, 0, 2)
                    p.background_color = (0, 0, 0)
                    # Note: taking negative angle here, as blender version rotates the camera, while pyvista rotates the obj
                    rotation = transformations.rotation_matrix(-angle, [0, 1, 0])
                    mesh_copy = mesh.copy()
                    mesh_copy.apply_transform(rotation)
                    p.add_mesh(mesh_copy)
                    rendered_mesh = p.screenshot(transparent_background=True)
                    del mesh_copy

                save_img(rendered_mesh, rendered_mesh_path)
