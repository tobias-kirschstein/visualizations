import os
import subprocess
import uuid
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import trimesh
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Pose
from elias.util.io import load_img

from visualizations.env import REPO_ROOT
from visualizations.math.vector import Vec3

"""
To create video:
ffmpeg -framerate 25 -i .\frame_%05d.png -pix_fmt yuv420p out.mp4
"""


def render_single_mesh(mesh_obj: Union[str, Path, trimesh.Trimesh],
                       image_width: int = 1024,
                       image_height: int = 1024,
                       camera_distance: float = 3.,
                       angle: float = 0,
                       crop_y_min: Optional[float] = None,
                       scale: Optional[Union[float, str]] = None,
                       location: Vec3 = Vec3(),
                       light_x: float = 1.6,
                       light_y: float = 2.2,
                       light_z: float = 8.2,
                       mirror_light_x: bool = False,
                       mirror_light_z: bool = False,
                       color: Tuple[float, float, float] = (0.8, 0.9, 1),
                       use_vertex_colors: bool = False,
                       use_vertex_alpha: bool = False,
                       use_orthographic_cam: bool = False,
                       fov: float = 0.6911110281944275,
                       cam_pose: Optional[Pose] = None,
                       ) -> np.ndarray:
    # Could do some complicated piping here, but no sure how to create mesh obj in bpy from binary .ply
    # blenderproc_cmd = subprocess.Popen(["blenderproc", "run", f"{REPO_ROOT}/scripts/render/single_mesh.py", "pipe", "out.img"], stdin=subprocess.PIPE)
    # blenderproc_cmd.stdin.write(b"hello")
    # result = blenderproc_cmd.communicate()

    random_id = uuid.uuid4().hex
    if use_vertex_alpha:
        # .obj is faster, but vertex alpha does not work with .obj
        tmp_input_file = f"{REPO_ROOT}/scripts/render/tmp_input_{random_id}.ply"
    else:
        tmp_input_file = f"{REPO_ROOT}/scripts/render/tmp_input_{random_id}.obj"

    tmp_output_file = f"{REPO_ROOT}/scripts/render/tmp_output_{random_id}.png"

    delete_temp_input = True
    try:
        if isinstance(mesh_obj, str) or isinstance(mesh_obj, Path):
            tmp_input_file = str(mesh_obj)
            assert Path(tmp_input_file).exists(), f"{tmp_input_file} does not exist"
            delete_temp_input = False
        else:
            scene = trimesh.Scene()
            scene.add_geometry(mesh_obj)
            scene.export(tmp_input_file)

        cmd_arguments = ["blenderproc",
                         "run",
                         f"{REPO_ROOT}/scripts/render/single_mesh.py",
                         tmp_input_file,
                         tmp_output_file,
                         "--image_width", str(image_width),
                         "--image_height", str(image_height),
                         "--camera_distance", str(camera_distance),
                         "--angle", str(angle),
                         "--crop_y_min", str(crop_y_min),
                         "--scale", str(scale),
                         "--location_x", str(location.x),
                         "--location_y", str(location.y),
                         "--location_z", str(location.z),
                         "--light_x", str(light_x),
                         "--light_y", str(light_y),
                         "--light_z", str(light_z),
                         "--color_r", str(color[0]),
                         "--color_g", str(color[1]),
                         "--color_b", str(color[2]),
                         "--fov", str(fov)
                         ]
        if use_vertex_colors:
            cmd_arguments.append("--use_vertex_colors")

        if use_vertex_alpha:
            cmd_arguments.append("--use_vertex_alpha")

        if not use_orthographic_cam:
            cmd_arguments.append("--no-use-orthographic-cam")

        if mirror_light_x:
            cmd_arguments.append("--mirror_light_x")

        if mirror_light_z:
            cmd_arguments.append("--mirror_light_z")

        if cam_pose is not None:
            cam_pose = cam_pose.change_pose_type(PoseType.CAM_2_WORLD, inplace=False)
            cam_pose = cam_pose.change_camera_coordinate_convention(CameraCoordinateConvention.OPEN_GL, inplace=False)
            cam_position = cam_pose.get_translation()
            cam_euler = cam_pose.get_euler_angles('xyz')
            cmd_arguments.extend([
                "--cam_translation_x", f"{cam_position.x: 0.9f}",
                "--cam_translation_y", f"{cam_position.y: 0.9f}",
                "--cam_translation_z", f"{cam_position.z: 0.9f}",
                "--cam_euler_x", f"{cam_euler.x: 0.9f}",
                "--cam_euler_y", f"{cam_euler.y: 0.9f}",
                "--cam_euler_z", f"{cam_euler.z: 0.9f}",
            ])

        blenderproc_cmd = subprocess.Popen(cmd_arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = blenderproc_cmd.communicate()

        if not Path(tmp_output_file).exists():
            print("RENDERING FAILED")
            print(out.decode())
            print(err.decode())
        rendered_img = load_img(tmp_output_file)

    except KeyboardInterrupt as e:
        raise e
    finally:
        # Ensure that temp files are always deleted, even if program is interrupted
        if delete_temp_input and Path(tmp_input_file).exists():
            os.remove(tmp_input_file)
        if Path(tmp_output_file).exists():
            os.remove(tmp_output_file)

    return rendered_img
