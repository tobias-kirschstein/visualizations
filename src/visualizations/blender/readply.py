import sys
from typing import Optional

import bpy

from visualizations.env import REPO_ROOT

sys.path.append(f"{REPO_ROOT}/submodules/blender-ply-import")
from readply import readply  # TODO: Doesn't work because we cannot compilein blender's bullshit python...


def blender_readply(file_path: str, scale: Optional[float] = None):
    p = readply(file_path)

    mesh = bpy.data.meshes.new(name='imported mesh')

    mesh.vertices.add(p['num_vertices'])
    mesh.vertices.foreach_set('co', p['vertices'])

    mesh.loops.add(len(p['faces']))
    mesh.loops.foreach_set('vertex_index', p['faces'])

    mesh.polygons.add(p['num_faces'])
    mesh.polygons.foreach_set('loop_start', p['loop_start'])
    mesh.polygons.foreach_set('loop_total', p['loop_length'])

    if 'vertex_normals' in p:
        mesh.vertices.foreach_set('normal', p['vertex_normals'])

    if 'vertex_colors' in p:
        vcol_layer = mesh.vertex_colors.new()
        vcol_data = vcol_layer.data
        vcol_data.foreach_set('color', p['vertex_colors'])

    if 'texture_coordinates' in p:
        uv_layer = mesh.uv_layers.new(name='default')
        uv_layer.data.foreach_set('uv', p['texture_coordinates'])

    mesh.validate()
    mesh.update()

    # Create object to link to mesh

    obj = bpy.data.objects.new('imported object', mesh)

    # Add object to the scene
    scene = bpy.context.scene
    scene.collection.children[0].objects.link(obj)

    # Select the new object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if scale is not None:
        bpy.ops.transform.resize(value=(scale, scale, scale))

    return obj
