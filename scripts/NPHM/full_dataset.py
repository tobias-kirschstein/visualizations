import os.path
from collections import defaultdict
from pathlib import Path

from cairo import ImageSurface, FORMAT_ARGB32, Context
from elias.util.io import save_img, load_img
from tqdm import tqdm

from visualizations.cairo.image import draw_image, to_image
from visualizations.env_nphm import NPHM_DATA_PATH
from visualizations.math.vector import Vec2
from visualizations.render.single_mesh import render_single_mesh

NPHM_ALL_SCANS_FOLDER = "//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/scans_processed"
NPHM_FULL_DATASET_RENDERING_FOLDER = f"{NPHM_DATA_PATH}/full_dataset"

RESOLUTION = 2048
IMAGE_HEIGHT = RESOLUTION
IMAGE_WIDTH = RESOLUTION

FINAL_IMAGE_WIDTH = 20000
OVERLAP_X = 0.1  # percentage of single portrait width
OVERLAP_Y = 0.1  # percentage of single portrait height
N_COLUMNS = 40

if __name__ == '__main__':
    scan_paths = Path(NPHM_ALL_SCANS_FOLDER).rglob('target.ply')

    scans_metadata = defaultdict(lambda: dict())
    n_total_scans = 0

    # Render blender meshes
    for scan_path in tqdm(scan_paths, desc="Collecting scans & blender rendering"):
        relative_path = os.path.relpath(scan_path, NPHM_ALL_SCANS_FOLDER)
        identity = Path(relative_path).parts[0]
        scan_name = Path(relative_path).parts[1]
        rendering_output_path = f"{NPHM_FULL_DATASET_RENDERING_FOLDER}/{identity}/{scan_name}.png"

        scans_metadata[identity][scan_name] = rendering_output_path
        n_total_scans += 1

        if not Path(rendering_output_path).exists():
            rendered_mesh = render_single_mesh(scan_path,
                                               image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                               crop_y_min=-17,
                                               scale=1 / 6)

            save_img(rendered_mesh, rendering_output_path)

    # Make portrait composition

    n_rows = int(n_total_scans / N_COLUMNS)
    portrait_aspect_ratio = IMAGE_HEIGHT / IMAGE_WIDTH

    column_width = FINAL_IMAGE_WIDTH / N_COLUMNS
    row_height = column_width * portrait_aspect_ratio

    portrait_width = column_width * (1 + OVERLAP_X)
    portrait_height = row_height * (1 + OVERLAP_Y)
    portrait_size = Vec2(portrait_width, portrait_height)
    final_image_height = int(n_rows * row_height)
    surface = ImageSurface(FORMAT_ARGB32, FINAL_IMAGE_WIDTH, final_image_height)
    ctx = Context(surface)

    rendering_paths = [rendering_path
                       for scan_metadata in scans_metadata.values()
                       for rendering_path in scan_metadata.values()]
    rendering_paths_iterator = iter(rendering_paths)
    for row in tqdm(range(n_rows), desc="Creating final image"):
        for col in range(N_COLUMNS):
            rendering_path = next(rendering_paths_iterator)
            portrait = load_img(rendering_path)
            center = Vec2(col * column_width + column_width / 2,
                          row * row_height + row_height / 2)
            draw_image(ctx, portrait, center, size=portrait_size)

    full_dataset_img = to_image(surface)
    save_img(full_dataset_img, f"{NPHM_FULL_DATASET_RENDERING_FOLDER}/full_dataset.png")
