from pathlib import Path

import trimesh
from elias.util.io import save_img
from tqdm import tqdm

from visualizations.env_nphm import NPHM_DATA_PATH
import pyvista as pv

from visualizations.render.single_mesh import render_single_mesh

NPHM_EXPRESSION_INTERPOLATIONS_FOLDER = f"//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/expression_interpol_test"
NPHM_EXPRESSION_INTERPOLATIONS_RENDERINGS_FOLDER = f"{NPHM_DATA_PATH}/expression_interpolations/renderings"

SELECTED_IDENTITES = ["mustafa_corrected"]

RESOLUTION = 1024
IMAGE_WIDTH = int(3/4 * RESOLUTION)
IMAGE_HEIGHT = RESOLUTION

USE_BLENDER = True


def create_renderings(run_name: str):
    mesh_paths = sorted(Path(f"{NPHM_EXPRESSION_INTERPOLATIONS_FOLDER}/{run_name}").glob("mesh_*.ply"))
    for frame_id, mesh_path in tqdm(enumerate(mesh_paths)):

        rendered_mesh_path = f"{NPHM_EXPRESSION_INTERPOLATIONS_RENDERINGS_FOLDER}/{run_name}/frame_{frame_id:05d}.png"
        if not Path(rendered_mesh_path).exists():
            if USE_BLENDER:
                rendered_mesh = render_single_mesh(mesh_path,
                                                   image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                                   use_orthographic_cam=False,
                                                   camera_distance=2,
                                                   mirror_light_x=True,
                                                   mirror_light_z=False)
            else:
                p = pv.Plotter(off_screen=True)
                mesh = trimesh.load(mesh_path)
                p.camera_set = True
                p.camera_position = 'xy'
                p.camera.position = (0, 0, 3)
                p.add_mesh(mesh)
                rendered_mesh = p.screenshot(transparent_background=True)
                del mesh
                p.clear()
                p.deep_clean()

            save_img(rendered_mesh, rendered_mesh_path)


if __name__ == '__main__':
    for run_folder in SELECTED_IDENTITES:
        create_renderings(run_folder)
