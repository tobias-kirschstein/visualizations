import os
import subprocess
from pathlib import Path
from time import sleep
from typing import Optional, Union

import numpy as np
import trimesh
from elias.util.io import load_img

from visualizations.env import REPO_ROOT


def render_single_mesh(mesh_obj: Union[str, Path, trimesh.Trimesh],
                       image_width: int = 1024,
                       image_height: int = 1024,
                       camera_distance: int = 3,
                       crop_y_min: Optional[float] = None,
                       scale: Optional[Union[float, str]] = None
                       ) -> np.ndarray:
    # Could do some complicated piping here, but no sure how to create mesh obj in bpy from binary .ply
    # blenderproc_cmd = subprocess.Popen(["blenderproc", "run", f"{REPO_ROOT}/scripts/render/single_mesh.py", "pipe", "out.img"], stdin=subprocess.PIPE)
    # blenderproc_cmd.stdin.write(b"hello")
    # result = blenderproc_cmd.communicate()

    tmp_input_file = f"{REPO_ROOT}/scripts/render/tmp_input.ply"
    tmp_output_file = f"{REPO_ROOT}/scripts/render/tmp_output.png"

    delete_temp_input = True
    if isinstance(mesh_obj, str) or isinstance(mesh_obj, Path):
        tmp_input_file = str(mesh_obj)
        delete_temp_input = False
    else:
        scene = trimesh.Scene()
        scene.add_geometry(mesh_obj)
        scene.export(tmp_input_file)

    blenderproc_cmd = subprocess.Popen(["blenderproc",
                                        "run",
                                        f"{REPO_ROOT}/scripts/render/single_mesh.py",
                                        tmp_input_file,
                                        tmp_output_file,
                                        "--image_width", str(image_width),
                                        "--image_height", str(image_height),
                                        "--camera_distance", str(camera_distance),
                                        "--crop_y_min", str(crop_y_min),
                                        "--scale", str(scale)
                                        ])
    blenderproc_cmd.communicate()

    rendered_img = load_img(tmp_output_file)

    if delete_temp_input:
        os.remove(tmp_input_file)
    os.remove(tmp_output_file)

    return rendered_img
