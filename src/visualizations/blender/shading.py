from typing import Union, Tuple, Optional

import bpy
from blenderproc.python.types.EntityUtility import Entity
from bpy.types import Object, Node, Material


def create_principled_bsdf_material(
        name: str,
        color: Optional[Tuple[float, float, float]] = None,
        metallic: Optional[float] = None,
        roughness: Optional[float] = None) -> Material:
    # if isinstance(obj, Entity):
    #     obj = obj.blender_obj

    material = bpy.data.materials.new(f'{name}_material')
    material.use_nodes = True

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')

    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Change color
    if color is not None:
        bsdf_node.inputs[0].default_value = (color[0], color[1], color[2], 1)

    # Change metallic
    if metallic is not None:
        bsdf_node.inputs[6].default_value = metallic

    # Specularity defaults to 0.5, we don't want that
    bsdf_node.inputs[7].default_value = 0

    # Change roughness
    if roughness is not None:
        bsdf_node.inputs[9].default_value = roughness

    # obj.data.materials.append(material)

    return material


def apply_material_to_obj(obj: Union[Entity, Object], material: Material):
    if isinstance(obj, Entity):
        obj = obj.blender_obj

    obj.data.materials.append(material)
