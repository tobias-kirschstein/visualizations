from unittest import TestCase

import trimesh
from matplotlib import pyplot as plt

from visualizations.render.single_mesh import render_single_mesh


class TestRenderSingleMesh(TestCase):

    def test_simple_trimesh(self):
        box = trimesh.creation.box(extents=[0.5, 1, 2])

        img = render_single_mesh(box, scale=1, angle=0.5, image_width=512, camera_distance=4)
        print(img.shape)

        plt.figure()
        plt.imshow(img)
        plt.show()