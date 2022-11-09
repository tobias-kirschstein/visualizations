from unittest import TestCase

import trimesh

from visualizations.render.single_mesh import render_single_mesh


class TestRenderSingleMesh(TestCase):

    def test_simple_trimesh(self):
        box = trimesh.creation.box(extents=[1, 1, 1])

        img = render_single_mesh(box, scale=3, image_width=512, camera_distance=4)
        print(img.shape)