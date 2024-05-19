from typing import Union, Tuple, Optional

import bpy
from blenderproc.python.types.EntityUtility import Entity
from bpy.types import Object, Node, Material


def create_principled_bsdf_material(
        name: str,
        color: Optional[Tuple[float, float, float]] = None,
        use_vertex_color: bool = False,
        use_vertex_alpha: bool = False,
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

    # Change color
    vertex_color_node = None
    if use_vertex_color:
        vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
        links.new(vertex_color_node.outputs['Color'],  bsdf_node.inputs['Base Color'])
    elif color is not None:
        bsdf_node.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1)

    if use_vertex_alpha:
        material.blend_method = 'BLEND'
        if vertex_color_node is None:
            vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
        mix_shader_node = nodes.new(type='ShaderNodeMixShader')
        transparent_bsdf_node = nodes.new(type='ShaderNodeBsdfTransparent')

        links.new(vertex_color_node.outputs['Alpha'], mix_shader_node.inputs['Fac'])
        links.new(transparent_bsdf_node.outputs['BSDF'], mix_shader_node.inputs[1])
        links.new(bsdf_node.outputs['BSDF'], mix_shader_node.inputs[2])
        links.new(mix_shader_node.outputs['Shader'], output_node.inputs['Surface'])
    else:
        links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

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
