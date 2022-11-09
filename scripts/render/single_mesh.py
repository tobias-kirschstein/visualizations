import blenderproc as bproc
# bproc import has to stay on top!
import tyro

from elias.util import ensure_directory_exists_for_file
from elias.util.io import save_img
from tyro.conf import Positional

from visualizations.blender.geometry import delete_vertices
from visualizations.blender.io import import_ply
from visualizations.blender.shading import create_principled_bsdf_material, \
    apply_material_to_obj
from typing import Optional, Union
import bpy

from visualizations.math.matrix import Pose

MAX_SAMPLES_PER_PIXEL = 20  # Ray-tracing render quality. Higher is better, but slower


def main(input_path: Positional[str],
         output_path: Positional[str],
         image_width: int = 1024,
         image_height: int = 1024,
         camera_distance: int = 3,
         crop_y_min: Optional[float] = None,
         scale: Optional[Union[float, str]] = None):
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
    :param crop_y_min:
        If specifies, crops all vertices with smaller y coordinate from the mesh before rendering
    :param scale:
        Scales the mesh before rendering
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

    bpy.context.scene.camera.data.type = "ORTHO"
    cam_to_world.move(z=camera_distance)
    bproc.camera.set_resolution(image_width, image_height)

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

    material = create_principled_bsdf_material("metallic", color=(0.8, 0.9, 1), metallic=0.9, roughness=0.7)
    obj = import_ply(input_path, scale=scale)

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
