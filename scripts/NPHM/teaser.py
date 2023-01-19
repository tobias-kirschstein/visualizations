

import blenderproc as bproc
# bproc import has to stay on top!

from math import sqrt, ceil
from typing import Literal
import os
from pathlib import Path

import numpy as np
from elias.util.io import save_img

from visualizations.blender.geometry import delete_vertices
from visualizations.blender.io import import_ply
from visualizations.blender.shading import create_principled_bsdf_material, \
    apply_material_to_obj
from visualizations.env import BLENDER_OUTPUT_PATH, REPO_ROOT
from elias.util.fs import ensure_directory_exists

import bpy

from visualizations.math.matrix import Pose

LAYOUT_TYPE: Literal['SQUARE', 'ROW'] = 'ROW'
ROW_STACKED = False  # If true, heads will be rendered perspectively 2x2 instead of 4 in a row

NPHM_ALL_SCANS_PATH = "D:/Projects/NPHM/data/best_scans"
CHOSEN_SCANS = ["angry_m", "barbara_smile_hair", "evgeni_brow_raiser", "moutWeird_f", "id73_schmollen_f", "id29_underweight_interesting_m", "david_smile"]

BODY_CROPY = -17  # Vertices with y-coordinates below that will be cropped
N_HEADS = 2  # How many heads the teaser should contain
MAX_SAMPLES_PER_PIXEL = 20  # Ray-tracing render quality. Higher is better, but slower

RESOLUTION = 1024
SQUARE_IMAGE_WIDTH = RESOLUTION
SQUARE_IMAGE_HEIGHT = RESOLUTION
SQUARE_OFFSET_X = 1.2 * 2
SQUARE_OFFSET_Y = 1.2 * 2
SQUARE_CAMERA_DISTANCE = 3

ROW_IMAGE_WIDTH = RESOLUTION
ROW_IMAGE_HEIGHT = int(0.6 * RESOLUTION)
ROW_OFFSET_X = 1.2
ROW_OFFSET_Y = 1.2 * 2 # Only for stacked
ROW_CAMERA_DISTANCE = 7
ROW_IMAGE_HEIGHT_STACKED = RESOLUTION
ROW_IMAGE_WIDTH_STACKED = int(3/4 * RESOLUTION)
ROW_OFFSET_X_STACKED = 1.4

# ----------------------------------------------------------
# bproc scene setup
# ----------------------------------------------------------

bproc.init()
bproc.renderer.enable_depth_output(False)
bproc.renderer.set_max_amount_of_samples(MAX_SAMPLES_PER_PIXEL)

# Make background transparent
bpy.context.scene.render.image_settings.color_mode = 'RGBA'
bpy.context.scene.render.film_transparent = True
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 0)

# ----------------------------------------------------------
# Cameras
# ----------------------------------------------------------
cam_to_world = Pose()

if LAYOUT_TYPE == 'SQUARE':
    bpy.context.scene.camera.data.type = "ORTHO"
    cam_to_world.move(z=SQUARE_CAMERA_DISTANCE)
    bproc.camera.set_resolution(SQUARE_IMAGE_WIDTH, SQUARE_IMAGE_HEIGHT)
elif LAYOUT_TYPE == 'ROW':
    cam_to_world.move(z=ROW_CAMERA_DISTANCE)
    if ROW_STACKED:
        bproc.camera.set_resolution(ROW_IMAGE_WIDTH_STACKED, ROW_IMAGE_HEIGHT_STACKED)
    else:
        bproc.camera.set_resolution(ROW_IMAGE_WIDTH, ROW_IMAGE_HEIGHT)

bproc.camera.add_camera_pose(cam_to_world)

# ----------------------------------------------------------
# Lighting Setup
# ----------------------------------------------------------

light = bproc.types.Light()
light.set_location([1.6, 2.2, 8.2])
light.set_type("AREA")
light.set_energy(1000)  # Energy is Watts
light.blender_obj.data.shadow_soft_size = 2

# ----------------------------------------------------------
# Load Meshes
# ----------------------------------------------------------

material = create_principled_bsdf_material("head", color=(0.8, 0.9, 1), metallic=0.9, roughness=0.7)

head_meshes = []
nrows = ceil(sqrt(N_HEADS))
for i_head, mesh_name in enumerate(CHOSEN_SCANS):
    mesh_file = f"{NPHM_ALL_SCANS_PATH}/{mesh_name}.ply"
    head_mesh = import_ply(mesh_file, scale=1 / 12.5)

    delete_vertices(head_mesh, lambda v: v.co.y < BODY_CROPY)

    apply_material_to_obj(head_mesh, material)

    if LAYOUT_TYPE == 'SQUARE':
        head_mesh.location.x = SQUARE_OFFSET_X * (i_head % nrows - (nrows - 1) / 2)
        head_mesh.location.y = SQUARE_OFFSET_Y * (-int(i_head / nrows) + (nrows - 1) / 2)
        head_mesh.location.z = i_head * 0.1
    elif LAYOUT_TYPE == 'ROW':
        if ROW_STACKED:
            head_mesh.location.z = -i_head * 1.5 + int(i_head / nrows) * 2.2 * 1.5
            head_mesh.location.x = ((i_head % nrows) - 0.25) * ROW_OFFSET_X_STACKED
            head_mesh.location.y = (-int(i_head / nrows) + (nrows - 1) / 2) * ROW_OFFSET_Y
        else:
            head_mesh.location.z = -i_head
            head_mesh.location.x = (i_head - 1) * ROW_OFFSET_X
        head_mesh.rotation_euler[1] = 1/4 * np.pi

    head_meshes.append(head_mesh)

    if len(head_meshes) >= N_HEADS:
        break

# ----------------------------------------------------------
# Rendering
# ----------------------------------------------------------

data = bproc.renderer.render()
rendered_image = data['colors'][0]

# ----------------------------------------------------------
# Saving images
# ----------------------------------------------------------

# Store blender file in blender_output/.../script_name
relative_script_path = Path(os.path.relpath(__file__, f"{REPO_ROOT}/scripts"))
relative_folder_path = '/'.join(relative_script_path.parts[:-1])
output_directory = f"{BLENDER_OUTPUT_PATH}/{relative_folder_path}"
ensure_directory_exists(output_directory)
blender_file_output_path = f"{output_directory}/{relative_script_path.stem}/main_file.blend"

save_img(rendered_image, f"{output_directory}/{relative_script_path.stem}/rendering.png")
bpy.ops.wm.save_mainfile(filepath=blender_file_output_path)
