from pathlib import Path
from typing import Union

import cairo
import numpy as np
import pyvista as pv
import trimesh
from cairo import Format
from elias.util import ensure_directory_exists_for_file
from elias.util.io import save_img, load_img
from matplotlib import pyplot as plt
from scipy.spatial.transform.rotation import Rotation as R

from visualizations.cairo.image import draw_image, to_image
from visualizations.env_nphm import NPHM_DATA_PATH
from visualizations.math.vector import Vec2
from visualizations.render.single_mesh import render_single_mesh

NEUTRAL_SCAN_ID_MOUTH_OPEN = 10  # for NPM, NPHM
NEUTRAL_SCAN_ID_MOUTH_CLOSED = 0  # for FLAME, BFM
NEUTRAL_SCAN_NAME = "s_steffen_e"

NPHM_RESULTS_FOLDER = f"{NPHM_DATA_PATH}/results_expression"
METHODS_ORDER = ['PC', 'FLAME', 'localPCA', 'imface_star', 'NPM', 'NPHM', 'GT']
GRID_LINE_X_ADJUSTMENTS = [0.05, 0.02, 0.05, 0.02, 0.02, 0]  # Manual adjustments for vertical grid lines. Measured in percentage of cell width

SUPPLEMENTAL_METHODS_ORDER = ['PC', 'BFM', 'globalPCA', 'ImFace', 'imface_star', 'NPHM', 'GT']
SUPPLEMENTAL_GRID_LINE_X_ADJUSTMENTS = [0.05, 0.02, 0.02, 0.02, 0.02, 0]

USE_BLENDER = True
USE_SUPPLEMENTAL = True
SHOW_NEUTRAL_SCANS = False

RESOLUTION = 2048
IMAGE_WIDTH = int(0.6 * RESOLUTION)
IMAGE_HEIGHT = RESOLUTION

FIGURE_WIDTH = 2048
STRIPES_SMALL = 1.5  # Relative width of first column (pointcloud observation)
STRIPES_WIDE = 3  # Relative width of all other cells
PADDING_X = 0  # Padding between columns
PADDING_Y = -0.15  # Padding between rows
GLOBAL_OFFSET_Y = -0.03  # -0.05  # Move everything up a bit

# CHOSEN_FITTINGS = [1, 6, 18] #15] #, 18]
# CHOSEN_FITTINGS = [f"s_steffen_e_{expression_id}" for expression_id in [1, 15, 5, 9, 4, 8]]
CHOSEN_FITTINGS = [f"s_steffen_e_{expression_id}" for expression_id in [15, 9, 4]]
# CHOSEN_FITTINGS = [f"s_id88_e_{expression_id}" for expression_id in [10, 2, 8, 20, 19, 14]]

# [x, y, size] in percentage of render image width/height
CROP_EYE_LEFT = [0.27, 0.4, 0.2]
CROP_EYE_RIGHT = [0.53, 0.4, 0.2]
CROP_NOSE = [0.4, 0.45, 0.2]
CROP_MOUTH_LEFT = [0.3, 0.55, 0.2]
CROP_MOUTH_LEFT_LARGE = [0.35, 0.57, 0.25]
CROP_MOUTH_RIGHT_LARGE = [0.4, 0.57, 0.25]
CROP_CHIN = [0.36, 0.57, 0.28]
CROP_CHEEK_RIGHT = [0.52, 0.52, 0.2]

# x, y, size in percentage of render image width/height
# Steffen
CROPS = [
    # CROP_CHEEK_RIGHT,
    CROP_CHIN,
    # CROP_CHIN,
    CROP_MOUTH_RIGHT_LARGE,
    CROP_CHIN,
    # CROP_EYE_LEFT,
]
# id88
# CROPS = [
#     CROP_CHEEK_RIGHT,
#     CROP_CHIN,
#     CROP_CHIN,
#     CROP_MOUTH_RIGHT_LARGE,
#     CROP_EYE_LEFT,
#     CROP_CHEEK_RIGHT,
# ]
ERROR_NORMALIZATION = 0.020
LINE_COLOR = (0, 0, 1)
LINE_WIDTH_GRID = 2
LINE_WIDTH_CROPS = 1.5
CMAP_OFFSET = 0.05  # Ignore first percentages of cmap to get lighter colors for better visual inspection
CROPOUT_SIZE = 1.4
ERROR_MASK_SIZE = 1.2
COL_ERROR_BAR = 3  # In which column the error bar should be plotted
COLOR_BAR_HEIGHT = 0.9 * 0.5  # Percentage of cell height
COLOR_BAR_Y_OFFSET = 0.05  # Percentage of cell height

def load_mesh(method: str, fitting: str) -> Union[trimesh.Trimesh, np.ndarray]:
    if fitting == 'NEUTRAL':
        # GT and pointcloud shouldn't be shown, as it would be two different ones for FLAME / NPHM
        assert method not in {'GT', 'PC'}
        if method in {'FLAME', 'BFM'}:
            fitting = f"{NEUTRAL_SCAN_NAME}_{NEUTRAL_SCAN_ID_MOUTH_CLOSED}"
        else:
            fitting = f"{NEUTRAL_SCAN_NAME}_{NEUTRAL_SCAN_ID_MOUTH_OPEN}"

    if method == 'PC':
        # special handling for point clouds
        pc_path = f"{NPHM_RESULTS_FOLDER}/{method}/{fitting}.npy"
        points = np.load(pc_path)
        return points
    else:

        if method == 'GT':
            mesh_path = f"{NPHM_RESULTS_FOLDER}/{method}/{fitting}.ply"
            if not Path(mesh_path).exists():
                mesh_path = f"{NPHM_RESULTS_FOLDER}/{method}/mesh_{fitting}.ply"
        else:
            mesh_path = f"{NPHM_RESULTS_FOLDER}/{method}/mask_{fitting}.ply"
            if not Path(mesh_path).exists():
                mesh_path = f"{NPHM_RESULTS_FOLDER}/{method}/mesh_{fitting}.ply"
        # Need
        return trimesh.load(mesh_path, process=False)


def load_errors(method: str, fitting: str) -> np.ndarray:
    assert method not in {'PC', 'GT'}

    error_path = f"{NPHM_RESULTS_FOLDER}/{method}/error_{fitting}.npy"
    return np.load(error_path)


def load_error_mask(method: str, fitting: str) -> np.ndarray:
    assert method not in {'PC', 'GT'}

    error_mask_path = f"{NPHM_RESULTS_FOLDER}/{method}/mask_{fitting}.npy"
    return np.load(error_mask_path)


def scaled_turbo_cm(x: float):
    turbo_cm = plt.get_cmap('turbo')
    return turbo_cm((x + CMAP_OFFSET) / (1 + CMAP_OFFSET))


def render_color_bar() -> np.ndarray:
    p = pv.Plotter(off_screen=True)
    p.background_color = (0, 0, 0, 0)
    p.camera_set = True
    p.window_size = (140, 200)
    s_args = {'vertical': True,
              'n_labels': 3,
              'height': 0.8,
              'width': 0.8,
              'position_x': 0.05,
              'title': "Error [mm]",
              'fmt': ' %.1f',
              'label_font_size': 24,
              'title_font_size': 26,
              'color': 'black'
              }
    p.add_points(np.array([[-100, -100, -100], [-100, -100, -100]]),
                 point_size=0,
                 scalars=[0, ERROR_NORMALIZATION / 4 * 1000],  # Have to divide errors by 4 to get errors in [m]
                 scalar_bar_args=s_args,
                 cmap=scaled_turbo_cm)
    color_bar_img = p.screenshot(transparent_background=True)
    return color_bar_img


def scaled_cmap(cmap_name: str, scale: float):
    cmap = plt.get_cmap(cmap_name)
    return lambda x: cmap(x * scale)


if __name__ == '__main__':
    methods_order = SUPPLEMENTAL_METHODS_ORDER if USE_SUPPLEMENTAL else METHODS_ORDER
    grid_line_x_adjustments = SUPPLEMENTAL_GRID_LINE_X_ADJUSTMENTS if USE_SUPPLEMENTAL else GRID_LINE_X_ADJUSTMENTS

    # fittings = [gt_scan.stem for gt_scan in Path(f"{NPHM_RESULTS_FOLDER}/GT").iterdir()]
    # fittings = ['_'.join(fitting.split('_')[1:]) if fitting.startswith('mesh_') else fitting for fitting in fittings]
    # fittings = [fitting for i_fitting, fitting in enumerate(fittings) if i_fitting in CHOSEN_FITTINGS]
    fittings = CHOSEN_FITTINGS.copy()

    if SHOW_NEUTRAL_SCANS:
        fittings.insert(0, 'NEUTRAL')  # First row is neutral expressions

    plt.figure()
    nrows = len(fittings)
    ncols = len(methods_order)

    aspect_ratio = IMAGE_HEIGHT / IMAGE_WIDTH
    # width is divided into small column stripes. Small cells will be 2 stripes, wide cells will be 3 stripes
    n_total_stripes = STRIPES_SMALL + (
                ncols - 1) * STRIPES_WIDE  # All columns except point cloud have a small side column
    stripe_width = (FIGURE_WIDTH * (1 - ((ncols - 1)) * PADDING_X)) / n_total_stripes
    padding_width = FIGURE_WIDTH * PADDING_X

    cell_width_small = STRIPES_SMALL * stripe_width
    cell_width_wide = STRIPES_WIDE * stripe_width
    rendering_width = 2 * stripe_width
    cell_height = aspect_ratio * rendering_width
    padding_height = cell_height * PADDING_Y
    figure_height = int(nrows * cell_height + (nrows - 1) * padding_height)
    global_offset_y = GLOBAL_OFFSET_Y * figure_height
    figure_height = int(figure_height + global_offset_y)

    surface = cairo.ImageSurface(Format.ARGB32, FIGURE_WIDTH, figure_height)
    ctx = cairo.Context(surface)

    # Draw lines
    line_margin = (LINE_WIDTH_GRID - 1) / 2  # (0,0) -> 1,  (1, 1) -> 3, (2, 2) -> 5
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(LINE_WIDTH_GRID)
    ctx.move_to(line_margin, line_margin)
    ctx.line_to(FIGURE_WIDTH, line_margin)
    ctx.line_to(FIGURE_WIDTH, figure_height)
    ctx.line_to(line_margin, figure_height)
    ctx.line_to(line_margin, line_margin)
    ctx.stroke()
    for i_method in range(len(methods_order) - 1):
        line_x = cell_width_small + i_method * cell_width_wide + grid_line_x_adjustments[i_method] * cell_width_wide
        if i_method == COL_ERROR_BAR - 2:
            if SHOW_NEUTRAL_SCANS:
                ctx.move_to(line_x, 0)
                ctx.line_to(line_x, global_offset_y + cell_height + padding_height / 2 + COLOR_BAR_Y_OFFSET * cell_height)

                ctx.move_to(line_x, global_offset_y + COLOR_BAR_HEIGHT * cell_height + cell_height + padding_height / 2 + COLOR_BAR_Y_OFFSET * cell_height)
            else:
                ctx.move_to(line_x, global_offset_y + COLOR_BAR_HEIGHT * cell_height + COLOR_BAR_Y_OFFSET * cell_height)
        else:
            ctx.move_to(line_x, 0)
        ctx.line_to(line_x, figure_height)
        ctx.stroke()

    for i_fitting, fitting in enumerate(fittings):
        for i_method, method in enumerate(methods_order):

            # Don't show GT and pointcloud for neutral expression
            if fitting == 'NEUTRAL':
                if method in {'GT', 'PC'}:
                    continue

            mesh = load_mesh(method, fitting)

            p = pv.Plotter(off_screen=True)
            p.camera_set = True
            p.camera.position = (0, 0, 3)
            p.window_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
            # Necessary to set RGB values of transparent pixels to all 0s. Otherwise pycairo fucks up with alpha blending
            p.set_background('black')

            # Main Mesh / Pointcloud
            if method == 'PC':
                rotation_matrix_1 = R.from_euler(seq='xyz', angles=[0, -50 / 360 * 2 * np.pi, 0]).as_matrix()
                rotation_matrix_2 = R.from_euler(seq='xyz', angles=[0, np.pi / 4, 0]).as_matrix()
                # rotation_matrix = R.from_euler(seq='xyz', angles=[0, - np.pi / 6, 0]).as_matrix()
                pointcloud_canonical = mesh @ rotation_matrix_1.T
                pointcloud = pointcloud_canonical @ rotation_matrix_2.T
                p.theme.render_points_as_spheres = True
                p.add_points(pointcloud,
                             scalars=-pointcloud_canonical[:, 2],
                             cmap=scaled_cmap('turbo', 1),
                             clim=[-0.4, 0.4],
                             point_size=10)
                p.camera.position = (0, 0, 4)  # Move camera a bit left for pointcloud
                p.remove_scalar_bar()
                rendered_mesh = p.screenshot(transparent_background=True)
            else:
                if USE_BLENDER:
                    cached_rendering_path = f"{NPHM_RESULTS_FOLDER}/cache/{fitting}_{method}.png"
                    if Path(cached_rendering_path).exists():
                        rendered_mesh = load_img(cached_rendering_path)
                    else:
                        rendered_mesh = render_single_mesh(mesh,
                                                           image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                                           camera_distance=3, use_orthographic_cam=False, fov=np.pi / 6)
                        ensure_directory_exists_for_file(cached_rendering_path)
                        save_img(rendered_mesh, cached_rendering_path)
                else:
                    p.add_mesh(mesh)
                    rendered_mesh = p.screenshot(transparent_background=True)

            # Errors
            rendered_error_mesh = None
            if not method in {'PC', 'GT'} and not fitting == 'NEUTRAL':
                p = pv.Plotter(off_screen=True,
                               lighting='three lights')  # Make colors a big brighter by changing lighting

                p.set_background('black')
                p.camera_set = True
                p.camera.position = (0, 0, 3)
                p.window_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

                errors = load_errors(method, fitting)
                error_mask = load_error_mask(method, fitting)

                # Remove vertices AND faces that were filtered out during error calculation
                # The error is not calculated on the back of the head and for the hair
                mesh.remove_degenerate_faces()  # Important, otherwise mesh.vertex_faces fails
                faces_to_delete = np.unique(mesh.vertex_faces[~error_mask])[1:]
                faces_mask = np.ones((mesh.faces.shape[0]), dtype=np.bool)
                faces_mask[faces_to_delete] = False
                mesh.update_vertices(error_mask)
                mesh.update_faces(faces_mask)

                p.add_mesh(mesh, scalars=errors, cmap=scaled_turbo_cm, clim=[0, ERROR_NORMALIZATION])
                p.remove_scalar_bar()
                rendered_error_mesh = p.screenshot(transparent_background=True)

            # Plot main mesh
            if not fitting == 'NEUTRAL' or method not in {'PC', 'GT'}:
                if i_method == 0:
                    center_x = cell_width_small / 2
                elif i_method == len(methods_order) - 1:
                    center_x = cell_width_small + (ncols - 2) * cell_width_wide + rendering_width / 2
                else:
                    center_x = cell_width_small + (i_method - 1) * cell_width_wide + rendering_width / 2
                center_x += i_method * padding_width
                center_y = (i_fitting + 1) * cell_height - cell_height / 2 + (i_fitting) * padding_height + global_offset_y

                if method == 'PC':
                    center_x += stripe_width / 3  # Move pointcloud slightly more to the right

                center = Vec2(center_x, center_y)
                size = Vec2(rendering_width, cell_height)
                draw_image(ctx, rendered_mesh, center, size)

            # For actual methods plot an error mesh and a crop out
            if not method == 'PC' and not fitting == 'NEUTRAL':
                # Move smaller right images a bit more to the left to make use of empty space in renderings
                center_x = cell_width_small + i_method * cell_width_wide - 0.75 * stripe_width + i_method * padding_width
                center_y = i_fitting * cell_height + 0.2 * cell_height + (i_fitting) * padding_height + global_offset_y
                center = Vec2(center_x,
                              center_y + cell_height / 12)  # Move error masks down a little bit for compactness
                size = Vec2(stripe_width, cell_height / 2) * ERROR_MASK_SIZE

                # No error mesh for GT
                if not method == 'GT':
                    draw_image(ctx, rendered_error_mesh, center, size * 1.2)  # Make masks a tiny bit larger and overlap
                center_y += 0.05 * cell_height  # Add a little padding between error mask and cropout

                # Get crop outs
                if SHOW_NEUTRAL_SCANS:
                    crop_x, crop_y, crop_size = CROPS[i_fitting - 1]
                else:
                    crop_x, crop_y, crop_size = CROPS[i_fitting]
                crop_x = int(rendered_mesh.shape[1] * crop_x)
                crop_y = int(rendered_mesh.shape[0] * crop_y)
                crop_size = int(rendered_mesh.shape[1] * crop_size)

                # Find position to place cropout
                crop_out = rendered_mesh[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
                center_y += cell_height / 2
                center = Vec2(center_x, center_y)
                size = Vec2(stripe_width, stripe_width)  * CROPOUT_SIZE
                crop_out_rect_x = center.x - size.x / 2
                crop_out_rect_y = center.y - size.y / 2
                crop_out_size = size

                # Get original image crop positions
                shrink_factor = rendering_width / IMAGE_WIDTH  # Images in global coordinates are smaller than renderings
                crop_x_global = center_x - 1 / 4 * stripe_width - rendering_width + crop_x * shrink_factor
                crop_y_global = center_y - 3 / 4 * cell_height + crop_y * shrink_factor
                crop_size_global = crop_size * shrink_factor

                # Draw rectangle around source crop position
                ctx.set_source_rgb(*LINE_COLOR[::-1])
                ctx.set_line_width(LINE_WIDTH_CROPS)
                ctx.rectangle(crop_x_global, crop_y_global, crop_size_global, crop_size_global)
                ctx.stroke()

                # # Draw lines
                # ctx.save()
                # ctx.set_source_rgb(*LINE_COLOR[::-1])
                #
                # ctx.move_to(crop_x_global, crop_y_global)
                # ctx.line_to(crop_out_rect_x, crop_out_rect_y)
                #
                # ctx.move_to(crop_x_global + crop_size_global, crop_y_global)
                # ctx.line_to(crop_out_rect_x + crop_out_size.x, crop_out_rect_y)
                #
                # ctx.move_to(crop_x_global, crop_y_global + crop_size_global)
                # ctx.line_to(crop_out_rect_x, crop_out_rect_y + crop_out_size.y)
                #
                # ctx.move_to(crop_x_global + crop_size_global, crop_y_global + crop_size_global)
                # ctx.line_to(crop_out_rect_x + crop_out_size.x, crop_out_rect_y + crop_out_size.y)
                #
                # ctx.set_line_width(1)
                # ctx.set_dash([4, 6])
                # ctx.stroke()
                # ctx.restore()

                # Draw cropout
                draw_image(ctx, crop_out, center, size)

                # Draw rectangle around cropout
                ctx.set_source_rgb(*LINE_COLOR[::-1])
                ctx.set_line_width(LINE_WIDTH_CROPS)
                ctx.rectangle(crop_out_rect_x, crop_out_rect_y, crop_out_size.x, crop_out_size.y)
                ctx.stroke()

            # plt.subplot(nrows, ncols, i_fitting * ncols + i_method + 1)
            # plt.imshow(rendered_img)

    rendered_color_bar = render_color_bar()
    center_x = cell_width_small + (COL_ERROR_BAR - 2) * cell_width_wide +(COL_ERROR_BAR - 2) * padding_width + stripe_width / 8
    if SHOW_NEUTRAL_SCANS:
        center_y = cell_height + padding_height + 0.9 * cell_height / 3 + global_offset_y + COLOR_BAR_Y_OFFSET * cell_height
    else:
        center_y = COLOR_BAR_HEIGHT * cell_height / 2 + global_offset_y + COLOR_BAR_Y_OFFSET * cell_height
    draw_image(ctx, rendered_color_bar, Vec2(center_x, center_y), Vec2(stripe_width, COLOR_BAR_HEIGHT * cell_height))

    figure = to_image(surface)
    figure_name = "results_comparison_expression"
    if USE_SUPPLEMENTAL:
        figure_name += "_supplemental"
    save_img(figure, f"{NPHM_RESULTS_FOLDER}/{figure_name}.png")

    # Save jpeg with alpha channel replaced with white
    figure_jpg = figure[:, :, :3]
    alpha = figure[:, :, [3]] / 255
    figure_jpg = alpha * figure_jpg + (1 - alpha) * (np.ones_like(figure_jpg) * 255)
    save_img(figure_jpg.astype(np.uint8), f"{NPHM_RESULTS_FOLDER}/{figure_name}.jpg", jpg_quality=95)

    plt.figure()
    plt.imshow(figure)
    plt.show()
