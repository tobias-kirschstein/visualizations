from pathlib import Path

from elias.util.io import save_img

from visualizations.env_nphm import NPHM_DATA_PATH
from visualizations.render.single_mesh import render_single_mesh

NPHM_TEASER_PATH = f"{NPHM_DATA_PATH}/teaser"

RESOLUTION = 1024
IMAGE_WIDTH = int(RESOLUTION / 2)
IMAGE_HEIGHT = RESOLUTION

if __name__ == '__main__':
    mesh_paths = Path(NPHM_TEASER_PATH).glob("*.ply")
    for mesh_path in mesh_paths:
        rendered_image_path = f"{NPHM_TEASER_PATH}/{mesh_path.stem}.png"
        if not Path(rendered_image_path).exists():
            rendered_mesh = render_single_mesh(mesh_path, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, scale=4)
            save_img(rendered_mesh, rendered_image_path)