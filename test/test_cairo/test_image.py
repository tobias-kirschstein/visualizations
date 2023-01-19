from typing import Iterable
from unittest import TestCase

from cairo import Format, ImageSurface, Context

import test_cairo
import numpy as np

from visualizations.cairo.image import draw_image, to_image
from visualizations.math.vector import Vec2


class TestCairoImage(TestCase):

    def test_rgb(self):
        image_width = 11
        image_height = 13

        surface = ImageSurface(Format.RGB24, image_width, image_height)
        ctx = Context(surface)

        # Completely red rectangle
        red_image = np.zeros((5, 7, 3), dtype=np.uint8)
        red_image[:, :, 0] = 255

        draw_image(ctx, red_image, Vec2(2, 2))

        image_composite = to_image(surface)

        self.assertEqual(image_composite.shape[0], image_height)
        self.assertEqual(image_composite.shape[1], image_width)
        self.assertEqual(image_composite.shape[2], 3)

        self.assertEqual(image_composite[2, 2, 0], 255)
        self.assertEqual(image_composite[2, 2, 1], 0)
        self.assertEqual(image_composite[2, 2, 2], 0)

        # Completely green rectangle
        green_image = np.zeros((5, 7, 3), dtype=np.uint8)
        green_image[:, :, 1] = 255

        draw_image(ctx, green_image, Vec2(2, 2))

        image_composite = to_image(surface)

        self.assertEqual(image_composite.shape[0], image_height)
        self.assertEqual(image_composite.shape[1], image_width)
        self.assertEqual(image_composite.shape[2], 3)

        self.assertEqual(image_composite[2, 2, 0], 0)
        self.assertEqual(image_composite[2, 2, 1], 255)
        self.assertEqual(image_composite[2, 2, 2], 0)

        # Completely blue rectangle
        blue_image = np.zeros((5, 7, 3), dtype=np.uint8)
        blue_image[:, :, 2] = 255

        draw_image(ctx, blue_image, Vec2(2, 2))

        image_composite = to_image(surface)

        self.assertEqual(image_composite.shape[0], image_height)
        self.assertEqual(image_composite.shape[1], image_width)
        self.assertEqual(image_composite.shape[2], 3)

        self.assertEqual(image_composite[2, 2, 0], 0)
        self.assertEqual(image_composite[2, 2, 1], 0)
        self.assertEqual(image_composite[2, 2, 2], 255)

    def test_transparency(self):
        image_width = 11
        image_height = 13

        surface = ImageSurface(Format.ARGB32, image_width, image_height)
        ctx = Context(surface)

        half_transparent_red = np.zeros((6, 8, 4), dtype=np.uint8)
        half_transparent_red[:, :, 0] = 255
        half_transparent_red[:, :, 3] = 127

        half_transparent_green = np.zeros((6, 8, 4), dtype=np.uint8)
        half_transparent_green[:, :, 1] = 255
        half_transparent_green[:, :, 3] = 127

        draw_image(ctx, half_transparent_red, Vec2(2, 2))
        #image_composite = to_image(surface)
        #self._assertAllEqual(image_composite[2, 2], (255, 0, 0, 127))

        draw_image(ctx, half_transparent_green, Vec2(4, 4))

        image_composite = to_image(surface)

        # "A over B"
        color_a = np.array([0, 255, 0, 127]) / 255
        color_b = np.array([255, 0, 0, 127]) / 255
        alpha_a = color_a[3]
        alpha_b = color_b[3]
        alpha_result = alpha_a + (1 - alpha_a) * alpha_b
        color_result = 1 / alpha_result * (alpha_a * color_a[:3] + (1 - alpha_a) * alpha_b * color_b[:3])
        # TODO: For some reason, the green channel becomes pure 255, which is not what we would expect
        #   with "A over B" compositing
        self._assertAllEqual(image_composite[2, 2], (128, 255, 0, (127 + 255) / 2))

    def _assertAllEqual(self, first: np.ndarray, second: Iterable):
        self.assertTrue((first == second).all(), msg=f"{first} vs {second}")
