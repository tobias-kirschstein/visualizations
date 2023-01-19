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

NPHM_RESULTS_FOLDER = f"{NPHM_DATA_PATH}/results_identity"
METHODS_ORDER = ['PC', 'BFM', 'FLAME', 'NPM', 'NPHM', 'GT']
ERROR_MASK_MESH_FOLDER = [None, None, None, "_old", None, None]  # If face mesh for error mask differs from high-res mesh

USE_BLENDER = True

N_MAX_FITTINGS = 4

# CHOSEN_FITTINGS = [12, 17]  # 5, 13,
CHOSEN_FITTINGS = [5, 12, 13, 17]
RESOLUTION = 2048
IMAGE_WIDTH = int(0.6 * RESOLUTION)
IMAGE_HEIGHT = RESOLUTION

FIGURE_WIDTH = 2048
STRIPES_SMALL = 2
STRIPES_WIDE = 3
PADDING_X = 0.02  # Padding between columns
PADDING_Y = -0.15  # Padding between rows
GLOBAL_OFFSET_Y = -0.02  # Move everything up a bit

# [x, y, size] in percentage of render image width/height
CROP_EYE_LEFT = [0.27, 0.4, 0.2]
CROP_EYE_RIGHT = [0.53, 0.4, 0.2]
CROP_NOSE = [0.4, 0.45, 0.2]
CROP_MOUTH = [0.4, 0.59, 0.2]
CROP_CHIN = [0.4, 0.65, 0.2]

# CROPS = [
#     CROP_EYE_RIGHT,
#     CROP_EYE_RIGHT
# ]
CROPS = [
    CROP_MOUTH,
    CROP_EYE_RIGHT,
    CROP_EYE_LEFT,
    CROP_EYE_RIGHT,
]
ERROR_NORMALIZATION = 0.020
LINE_COLOR = (0, 0, 1)
CMAP_OFFSET = 0.05  # Ignore first percentages of cmap to get lighter colors for better visual inspection
CROPOUT_SIZE = 1.4
ERROR_MASK_SIZE = 1.2


def load_mesh(method: str, fitting: str, folder_suffix: str = "") -> Union[trimesh.Trimesh, np.ndarray]:
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
            mesh_path = f"{NPHM_RESULTS_FOLDER}/{method}{folder_suffix}/mask_{fitting}.ply"
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

    fittings = [gt_scan.stem for gt_scan in Path(f"{NPHM_RESULTS_FOLDER}/GT").iterdir()]
    fittings = ['_'.join(fitting.split('_')[1:]) if fitting.startswith('mesh_') else fitting for fitting in fittings]
    fittings = [fitting for i_fitting, fitting in enumerate(fittings) if i_fitting in CHOSEN_FITTINGS]
    fittings = fittings[:N_MAX_FITTINGS]
    # fittings = list(reversed(fittings))

    plt.figure()
    nrows = len(fittings)
    ncols = len(METHODS_ORDER)

    aspect_ratio = IMAGE_HEIGHT / IMAGE_WIDTH
    # width is divided into small column stripes. Small cells will be 2 stripes, wide cells will be 3 stripes
    n_total_stripes = STRIPES_SMALL + 5 * STRIPES_WIDE  # All columns except point cloud have a small side column
    stripe_width = (FIGURE_WIDTH * (1 - 5 * PADDING_X)) / n_total_stripes
    padding_width = FIGURE_WIDTH * PADDING_X

    cell_width_small = STRIPES_SMALL * stripe_width
    cell_width_wide = STRIPES_WIDE * stripe_width
    cell_height = aspect_ratio * cell_width_small
    padding_height = cell_height * PADDING_Y
    figure_height = int(nrows * cell_height + (nrows - 1) * padding_height)
    global_offset_y = GLOBAL_OFFSET_Y * figure_height

    surface = cairo.ImageSurface(Format.ARGB32, FIGURE_WIDTH, figure_height)
    ctx = cairo.Context(surface)

    for i_fitting, fitting in enumerate(fittings):
        for i_method, method in enumerate(METHODS_ORDER):
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
            if not method in {'PC', 'GT'}:
                p = pv.Plotter(off_screen=True,
                               lighting='three lights')  # Make colors a big brighter by changing lighting

                p.set_background('black')
                p.camera_set = True
                p.camera.position = (0, 0, 3)
                p.window_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

                errors = load_errors(method, fitting)
                error_mask = load_error_mask(method, fitting)

                if ERROR_MASK_MESH_FOLDER[i_method] is not None:
                    mesh = load_mesh(method, fitting, ERROR_MASK_MESH_FOLDER[i_method])

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
            if i_method == 0:
                center_x = cell_width_small / 2
            elif i_method == len(METHODS_ORDER) - 1:
                center_x = cell_width_small + 4 * cell_width_wide + cell_width_small / 2
            else:
                center_x = cell_width_small + (i_method - 1) * cell_width_wide + cell_width_small / 2
            center_x += i_method * padding_width
            center_y = (i_fitting + 1) * cell_height - cell_height / 2 + (i_fitting) * padding_height + global_offset_y

            if method == 'PC':
                center_x += stripe_width / 3  # Move pointcloud slightly more to the right

            center = Vec2(center_x, center_y)
            size = Vec2(cell_width_small, cell_height)
            draw_image(ctx, rendered_mesh, center, size)

            # For actual methods plot an error mesh and a crop out
            if not method == 'PC':
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
                crop_x, crop_y, crop_size = CROPS[i_fitting]
                crop_x = int(rendered_mesh.shape[1] * crop_x)
                crop_y = int(rendered_mesh.shape[0] * crop_y)
                crop_size = int(rendered_mesh.shape[1] * crop_size)

                # Find position to place cropout
                crop_out = rendered_mesh[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
                center_y += cell_height / 2
                center = Vec2(center_x, center_y)
                size = Vec2(stripe_width, stripe_width) * CROPOUT_SIZE
                crop_out_rect_x = center.x - size.x / 2
                crop_out_rect_y = center.y - size.y / 2
                crop_out_size = size

                # Get original image crop positions
                shrink_factor = cell_width_small / IMAGE_WIDTH  # Images in global coordinates are smaller than renderings
                crop_x_global = center_x - 1 / 4 * stripe_width - cell_width_small + crop_x * shrink_factor
                crop_y_global = center_y - 3 / 4 * cell_height + crop_y * shrink_factor
                crop_size_global = crop_size * shrink_factor

                # Draw rectangle around source crop position
                ctx.set_source_rgb(*LINE_COLOR[::-1])
                ctx.rectangle(crop_x_global, crop_y_global, crop_size_global, crop_size_global)
                ctx.stroke()

                # Draw lines
                ctx.save()
                ctx.set_source_rgb(*LINE_COLOR[::-1])

                ctx.move_to(crop_x_global, crop_y_global)
                ctx.line_to(crop_out_rect_x, crop_out_rect_y)

                ctx.move_to(crop_x_global + crop_size_global, crop_y_global)
                ctx.line_to(crop_out_rect_x + crop_out_size.x, crop_out_rect_y)

                ctx.move_to(crop_x_global, crop_y_global + crop_size_global)
                ctx.line_to(crop_out_rect_x, crop_out_rect_y + crop_out_size.y)

                ctx.move_to(crop_x_global + crop_size_global, crop_y_global + crop_size_global)
                ctx.line_to(crop_out_rect_x + crop_out_size.x, crop_out_rect_y + crop_out_size.y)

                ctx.set_line_width(1)
                ctx.set_dash([4, 6])
                ctx.stroke()
                ctx.restore()

                # Draw cropout
                draw_image(ctx, crop_out, center, size)

                # Draw rectangle around cropout
                ctx.set_source_rgb(*LINE_COLOR[::-1])
                ctx.rectangle(crop_out_rect_x, crop_out_rect_y, crop_out_size.x, crop_out_size.y)
                ctx.stroke()

            # plt.subplot(nrows, ncols, i_fitting * ncols + i_method + 1)
            # plt.imshow(rendered_img)

    rendered_color_bar = render_color_bar()
    center_x = cell_width_small + 4 * cell_width_wide + 4 * padding_width + stripe_width / 5
    center_y = 0.9 * cell_height / 3 + global_offset_y
    draw_image(ctx, rendered_color_bar, Vec2(center_x, center_y), Vec2(stripe_width, 0.9 * cell_height / 2))

    figure = to_image(surface)
    save_img(figure, f"{NPHM_RESULTS_FOLDER}/results_comparison_identity.png")
    save_img(figure[:, :, :3], f"{NPHM_RESULTS_FOLDER}/results_comparison_identity.jpg")

    plt.figure()
    plt.imshow(figure)
    plt.show()
