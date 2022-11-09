from typing import Callable

import bmesh
import bpy
from bpy.types import Object


def delete_vertices(obj: Object, predicate: Callable):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    vertices_to_delete = [v for v in bm.verts if predicate(v)]

    bmesh.ops.delete(bm, geom=vertices_to_delete)

    bmesh.update_edit_mesh(me)
    bpy.ops.object.mode_set(mode='OBJECT')
