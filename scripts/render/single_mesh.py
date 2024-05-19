import blenderproc as bproc
# bproc import has to stay on top!
import numpy as np
import tyro
from pathlib import Path
from elias.util import ensure_directory_exists_for_file
from elias.util.io import save_img
from tyro.conf import Positional

from visualizations.blender.geometry import delete_vertices
from visualizations.blender.io import import_ply
from visualizations.blender.shading import create_principled_bsdf_material, \
    apply_material_to_obj
from typing import Optional, Union
from math import cos, sin
import bpy

from visualizations.math.matrix import Pose, Intrinsics

MAX_SAMPLES_PER_PIXEL = 20  # Ray-tracing render quality. Higher is better, but slower


def main(input_path: Positional[str],
         output_path: Positional[str],
         image_width: int = 1024,
         image_height: int = 1024,
         camera_distance: float = 3.,
         angle: float = 0,
         cam_euler_x: Optional[float] = None,
         cam_euler_y: Optional[float] = None,
         cam_euler_z: Optional[float] = None,
         cam_translation_x: Optional[float] = None,
         cam_translation_y: Optional[float] = None,
         cam_translation_z: Optional[float] = None,
         crop_y_min: Optional[float] = None,
         scale: Optional[Union[float, str]] = None,
         location_x: float = 0,
         location_y: float = 0,
         location_z: float = 0,
         light_x: float = 1.6,
         light_y: float = 2.2,
         light_z: float = 8.2,
         color_r: float = 0.8,
         color_g: float = 0.9,
         color_b: float = 1,
         use_vertex_colors: bool = False,
         use_vertex_alpha: bool = False,
         fov: float = 0.6911110281944275,
         use_orthographic_cam: bool = True,
         mirror_light_x: bool= False,
         mirror_light_z: bool = False):
    """
    :param input_path:
        Path to the .ply file that shall be rendered. Choose "pipe" if mesh is directly input instead of read from file
    :param output_path:
        Path for the output image
    :param image_width:
        render width
    :param image_height:
        render height
    :param camera_distance:
        distance (in Blender meters) of the camera in z-direction from the origin
    :param fov:
        field of view of camera
    :param use_orthographic_cam:
        whether to use an orthographic projection instead of perspective
    :param angle:
        angle of the camera (in radians) around the y axis. Useful for rotating around the object
    :param crop_y_min:
        If specifies, crops all vertices with smaller y coordinate from the mesh before rendering
    :param scale:
        Scales the mesh before rendering
    :param location_x:
        Optional x offset for mesh (applied after scaling)
    :param location_y:
        Optional y offset for mesh (applied after scaling)
    :param location_z:
        Optional z offset for mesh (applied after scaling)

    :param light_x:
        x coordinate for light
    :param light_y:
        y coordinate for light
    :param light_z:
        How far the light should be away
    :param mirror_light_x:
        Creates duplicate(s) of the light source(s) that is mirrored in x-direction (flipped at the y-z plane)
    :param mirror_light_z:
        Creates duplicate(s) of the light source(s) that is mirrored in z-direction (flipped at the x-y plane)

    :param color_r:
        red channel for mesh. Only used if use_vertex_colors=False.
    :param color_g:
        green channel for mesh. Only used if use_vertex_colors=False.
    :param color_b:
        blue channel for mesh. Only used if use_vertex_colors=False.
    :param use_vertex_colors:
        If use_vertex_colors=True, the mesh will be rendered using the vertex colors from the .ply
    :param use_vertex_alpha:
        If specified, the mesh will be transparent in areas where the 4-th channel of the vertex color attribute has small values

    :return:
    """

    if isinstance(scale, str):
        scale = eval(scale)  # Potentially parse inputs like 1/25 to float

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

    # Assume y is up-direction
    if use_orthographic_cam:
        bpy.context.scene.camera.data.type = "ORTHO"
    if cam_translation_x is None and cam_translation_y is None and cam_translation_z is None:
        cam_z = camera_distance * cos(angle)
        cam_x = camera_distance * sin(angle)
        cam_to_world.move(x=cam_x, z=cam_z)
    else:
        cam_to_world.move(x=cam_translation_x or 0, y=cam_translation_y or 0, z=cam_translation_z or 0)

    if cam_euler_x is None and cam_euler_y is None and cam_euler_z is None:
        cam_to_world.rotate_euler('xyz', euler_y=angle)
    else:
        cam_to_world.rotate_euler('xyz', euler_x=cam_euler_x or 0, euler_y=cam_euler_y or 0, euler_z=cam_euler_z or 0)

    bproc.camera.set_resolution(image_width, image_height)
    bproc.camera.set_intrinsics_from_blender_params(fov, lens_unit='FOV')

    bproc.camera.add_camera_pose(cam_to_world)

    # ----------------------------------------------------------
    # Lighting Setup
    # ----------------------------------------------------------

    energy_front = 500 if mirror_light_x else 1000
    light = bproc.types.Light()
    light.set_location([light_x, light_y, light_z])
    light.set_type("AREA")
    light.set_energy(energy_front)  # Energy is Watts
    light.blender_obj.data.shadow_soft_size = 2

    if mirror_light_x:
        light_mirrored_x = bproc.types.Light()
        light_mirrored_x.set_location([-light_x, light_y, light_z])
        light_mirrored_x.set_type("AREA")
        light_mirrored_x.set_energy(energy_front)  # Energy is Watts
        light_mirrored_x.blender_obj.data.shadow_soft_size = 2

    if mirror_light_z:
        energy_back = 1000 if mirror_light_x else 2000
        light_mirrored_z = bproc.types.Light()
        light_mirrored_z.set_location([-2 * light_x, light_y, -light_z])
        light_mirrored_z.set_energy(energy_back)  # Energy is Watts
        light_mirrored_z.blender_obj.data.shadow_soft_size = 2

        if mirror_light_x:
            light_mirrored_x_z = bproc.types.Light()
            light_mirrored_x_z.set_location([2 * light_x, light_y, -light_z])
            light_mirrored_x_z.set_energy(energy_back)  # Energy is Watts
            light_mirrored_x_z.blender_obj.data.shadow_soft_size = 2

    # ----------------------------------------------------------
    # Load Meshes
    # ----------------------------------------------------------

    material = create_principled_bsdf_material("metallic",
                                               color=(color_r, color_g, color_b),
                                               metallic=0.9, roughness=0.7,
                                               use_vertex_color=use_vertex_colors,
                                               use_vertex_alpha=use_vertex_alpha)
    # obj = blender_readply(input_path, scale=scale)
    # obj = import_ply(input_path, scale=scale)

    if Path(input_path).suffix == '.obj':
        bproc.loader.load_obj(input_path, forward_axis='Y', up_axis='Z')  # Blender does some weird coordinate shuffling with .obj imports. Correct it here
        obj = bpy.context.active_object
    else:
        bproc.loader.load_obj(input_path)
        obj = bpy.context.active_object
        obj.data.materials.clear()  # load_obj() creates some default material that we do not need

    if scale is not None:
        bpy.ops.transform.resize(value=(scale, scale, scale))

    obj.location = (location_x, location_y, location_z)

    if crop_y_min is not None:
        delete_vertices(obj, lambda v: v.co.y < crop_y_min)

    apply_material_to_obj(obj, material)

    # ----------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------

    data = bproc.renderer.render()
    rendered_image = data['colors'][0]

    # ----------------------------------------------------------
    # Saving images
    # ----------------------------------------------------------

    ensure_directory_exists_for_file(output_path)
    save_img(rendered_image, output_path)
    # bpy.ops.wm.save_mainfile(filepath=blender_file_output_path)


if __name__ == '__main__':
    tyro.cli(main)
