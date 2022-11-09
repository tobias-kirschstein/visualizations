from pathlib import Path
from typing import Union, Optional

import bpy


def import_ply(path: Union[Path, str], scale: Optional[float] = None) -> bpy.types.Object:
    bpy.ops.import_mesh.ply(filepath=str(path))
    obj = bpy.context.active_object
    if scale is not None:
        bpy.ops.transform.resize(value=(scale, scale, scale))
    return obj
