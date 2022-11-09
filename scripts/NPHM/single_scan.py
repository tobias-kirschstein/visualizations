import blenderproc as bproc
# bproc import has to stay on top!

import os
from pathlib import Path
from typing import Union
from blenderproc.python.types.EntityUtility import Entity
from bpy.types import Object, Node
from elias.util.io import save_img

from visualizations.env import BLENDER_OUTPUT_PATH, REPO_ROOT
from elias.util.fs import ensure_directory_exists

import bpy

from visualizations.math.matrix import Pose

NPHM_SCANS_PATH = "D:/Projects/NPHM/data/best_scans"

MAX_SAMPLES_PER_PIXEL = 20  # Ray-tracing render quality. Higher is better, but slower
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 1024


def import_ply(path: str, scale: float = 1) -> bpy.types.Object:
    bpy.ops.import_mesh.ply(filepath=path)
    obj = bpy.context.active_object
    if not scale == 1:
        bpy.ops.transform.resize(value=(scale, scale, scale))
    return obj


# Durch 25 tilen!

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
cam_to_world.move(z=2)
bproc.camera.add_camera_pose(cam_to_world)

bproc.camera.set_resolution(IMAGE_WIDTH, IMAGE_HEIGHT)

# ----------------------------------------------------------
# Lighting Setup
# ----------------------------------------------------------

light = bproc.types.Light()
light.set_location([0.2, 1, 3])
light.set_type("POINT")
light.set_energy(1000)  # Energy is Watts
light.blender_obj.data.shadow_soft_size = 2

# ----------------------------------------------------------
# Load Meshes
# ----------------------------------------------------------

head_mesh = import_ply(f"{NPHM_SCANS_PATH}/barbara_smile_hair.ply", scale=1 / 25)


def create_principled_bsdf(obj: Union[Entity, Object], name: str) -> Node:
    if isinstance(obj, Entity):
        obj = obj.blender_obj

    material = bpy.data.materials.new(f'{name}_material')
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')

    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    obj.data.materials.append(material)

    return bsdf_node


bsdf_node = create_principled_bsdf(head_mesh, "head")

# Change color
bsdf_node.inputs[0].default_value = (0.0603202, 0.36982, 0.8, 1)

# Change metallic
bsdf_node.inputs[6].default_value = 0.9

# Change specularity
bsdf_node.inputs[7].default_value = 0

# Change roughness
bsdf_node.inputs[9].default_value = 0.7

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
