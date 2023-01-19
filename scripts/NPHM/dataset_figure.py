from pathlib import Path
from typing import Optional, Tuple
import pixie
import cairo
import numpy as np

from elias.util.io import load_img, save_img
from matplotlib import pyplot as plt

from visualizations.cairo.arrow import draw_arrow
from visualizations.cairo.image import draw_image, to_image
from visualizations.env_nphm import NPHM_DATA_PATH
from visualizations.math.vector import Vec2
from visualizations.render.single_mesh import render_single_mesh

"""
females

barbara
67
75		(Kopftuch)
86
silke


Have neutral mouth open:
-------------------------
id59
id67
id75	(Kopftuch + floater)
id78
id85
id86

expressions (as for id86):
1 - neutral mouth open
2 - smile
4 - dimpler
6 - shout
9 - angry
10 - mouth left
12 - kiss
14 - grin
15 - puff cheek
17 - sad
18 - wrinkles forehead
20 - extreme
21 - evil
"""

INTERESTING_EXPRESSIONS = [1, 2, 4, 6, 9, 10, 12, 14, 15, 17, 18, 20, 21]
SELECTED_EXPRESSIONS = [1, 9, 10, 14, 20]  # 4, 15

# CANDIDATE_IDENTITIES = ["silke", "evgeni", 59, 75, 78, 85, 86, 88, 92, 96, 97, 98]  # 67,
CANDIDATE_IDENTITIES = ["evgeni", 78, 85, 86, 88, 92, 96, 97, 98]  # 67,
N_IDENTITIES_MAX = 9

EXPRESSION_ID_CORRECTION = {
    "silke": {1: 10, 2: 1, 4: 3, 6: 5, 9: 8, 10: 9},
    "evgeni": {1: 10, 2: 1, 4: 3, 6: 5, 9: 8, 10: 9},
    59: {6: None, 21: None},
    67: {2: None, 4: 2, 6: 4, 9: 7, 10: 8, 12: 10, 14: 12, 15: None, 17: 14, 18: 15, 20: 17, 21: 18},
    78: {14: None, 15: 14, 17: 16, 18: 17, 20: 19, 21: 20},
    85: {2: None, 18: None, 20: 19, 21: 20},
    87: {1: None},  # bad
    88: {21: 20},
    89: {1: None},  # bad
    90: {1: None},  # not expressive
    92: {6: 5, 9: 8, 10: 9, 12: 11, 14: 13, 15: 14, 17: 16, 18: 17, 20: 19, 21: 20},
    93: {1: None},
    96: {6: 5, 9: 8, 10: 9, 12: 11, 14: 13, 15: 14, 17: 16, 18: 17, 20: 19, 21: 20},
    98: {1: 2, 2: 3, 4: 5, 6: 7, 9: 10, 10: 11, 12: 14, 14: 16, 15: 17, 17: 19, 18: 20, 20: 22, 21: 23}
}

USE_BLENDER = True

OVERVIEW_PATH = "D:/Projects/NPHM/data/overview"
SCANS_PATH = "//wsl.localhost/Ubuntu/mnt/rohan/cluster/daidalos/sgiebenhain/figure_dataset"
RESOLUTION = 1024
IMAGE_WIDTH = RESOLUTION
IMAGE_HEIGHT = RESOLUTION

FIGURE_WIDTH = 4096
# FIGURE_HEIGHT is calculated automatically based on the #identities
OVERLAP_X = 0.35  # percentage of overlap of images in x direction
OVERLAP_Y = 0.5
GAP_NEUTRAL = 0.03  # percentage of width that will be reserved for a gap between neutral pose and the rest
MARGIN = 0.05  # margin at all sides
CREATE_HIGHLIGHT = False
HIGHLIGHT_COLOR = (0.95, 0.8, 0.4)


def get_subject_expression_id(identity: int, global_expression_id: int) -> Optional[int]:
    if identity in EXPRESSION_ID_CORRECTION and global_expression_id in EXPRESSION_ID_CORRECTION[identity]:
        return EXPRESSION_ID_CORRECTION[identity][expression]
    else:
        return global_expression_id


def get_scan_path(identity_string: str, subject_expression_id: int) -> Tuple[
    Optional[str], Optional[float], Optional[float]]:
    scan_path_1 = f"{SCANS_PATH}/{identity_string}/expression_{subject_expression_id}/target.ply"
    if Path(scan_path_1).exists():
        return scan_path_1, 1 / 9 * 25 * 4, -17 / 100

    expression_folder = f"{SCANS_PATH}/{identity_string}/{identity_string}_{subject_expression_id}"
    if not Path(expression_folder).exists():
        return None, None, None
    intermediate_folder_name = next(Path(expression_folder).iterdir()).name
    scan_path = f"{expression_folder}/{intermediate_folder_name}/target.ply"
    if not Path(scan_path).exists():
        return None, None, None
    else:
        return scan_path, 1 / 9, -17


if __name__ == '__main__':
    identities = CANDIDATE_IDENTITIES
    identity_strings = [f"id{identity}" if isinstance(identity, int) else identity for identity in identities]
    expressions = SELECTED_EXPRESSIONS

    # Unfortunately, not all identities have all expressions (+ the IDs are sometimes shifted)
    # Double-check that selected identities have the requested expressions and drop identity if not
    filtered_identities = []
    filtered_identity_strings = []
    for identity, identity_string in zip(identities, identity_strings):
        keep_identity = True
        for expression in expressions:
            subject_expression_id = get_subject_expression_id(identity, expression)
            if USE_BLENDER:
                scan_path, _, _ = get_scan_path(identity_string, subject_expression_id)
                if subject_expression_id is None or scan_path is None:
                    keep_identity = False
                    print(f"Dropping identity {identity_string} as it is missing expression {expression}")
                    break
            else:
                if subject_expression_id is None or not Path(
                        f"{OVERVIEW_PATH}/{identity_string}_{subject_expression_id}.png").exists():
                    keep_identity = False
                    print(f"Dropping identity {identity_string} as it is missing expression {expression}")
                    break

        if keep_identity:
            filtered_identities.append(identity)
            filtered_identity_strings.append(identity_string)

    filtered_identities = filtered_identities[:N_IDENTITIES_MAX]
    nrows, ncols = len(filtered_identities), len(expressions)

    margin = FIGURE_WIDTH * MARGIN
    gap_width = FIGURE_WIDTH * GAP_NEUTRAL
    aspect_ratio = IMAGE_HEIGHT / IMAGE_WIDTH
    head_width = (FIGURE_WIDTH - gap_width - 2 * margin) / len(expressions) / (1 - OVERLAP_X)
    head_height = head_width * aspect_ratio
    figure_height = int(head_height * len(filtered_identities) * (1 - OVERLAP_Y) + 2 * margin)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, FIGURE_WIDTH, figure_height)
    ctx = cairo.Context(surface)

    if CREATE_HIGHLIGHT:
        ctx.set_source_rgb(*HIGHLIGHT_COLOR[::-1])
        ctx.rectangle(margin, margin, head_width, figure_height - 2 * margin)
        ctx.fill()

    plt.figure(figsize=(10, 16))
    for i_identy, identity in enumerate(filtered_identities):
        for i_expression, expression in enumerate(expressions):
            identity_string = filtered_identity_strings[i_identy]
            subject_expression_id = get_subject_expression_id(identity, expression)

            rendered_image_path = f"{NPHM_DATA_PATH}/dataset_figure/{identity_string}_{expression}.png"

            if USE_BLENDER:
                if Path(rendered_image_path).exists():
                    rendered_head = load_img(rendered_image_path)
                else:
                    scan_path, scale, crop_y_min = get_scan_path(identity_string, subject_expression_id)
                    rendered_head = render_single_mesh(scan_path,
                                                       crop_y_min=crop_y_min, scale=scale,
                                                       light_x=1.4, light_y=1.4, light_z=10,
                                                       image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
                    save_img(rendered_head, f"{NPHM_DATA_PATH}/dataset_figure/{identity_string}_{expression}.png")
            else:
                file_name = f"{OVERVIEW_PATH}/{identity_string}_{subject_expression_id}.png"
                rendered_head = load_img(file_name)

            i_image = i_identy * ncols + i_expression

            center_x = (i_expression * head_width + head_width / 2) * (1 - OVERLAP_X) + margin
            center_y = (i_identy * head_height + head_height / 2) * (1 - OVERLAP_Y) + margin

            if i_expression == 0:
                center_x += 5  # Improve centering just a tiny bit
            else:
                # All expressions except for neutral one will be moved to the right to create a visual gap
                # between neutral expression and the rest
                center_x += FIGURE_WIDTH * GAP_NEUTRAL

            center = Vec2(center_x, center_y)
            size = Vec2(head_width, head_height)
            draw_image(ctx, rendered_head, center, size=size)

    figure = to_image(surface)
    save_img(figure, f"{NPHM_DATA_PATH}/dataset_figure/dataset_figure.png")
