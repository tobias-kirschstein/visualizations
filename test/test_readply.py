from unittest import TestCase

from visualizations.env import REPO_ROOT


class ReadPLYTest(TestCase):
    def test_read_ply(self):
        from visualizations.blender.readply import readply

        result = readply(f"{REPO_ROOT}/test/test_assets/teapot.ply")
        self.assertEqual(result['num_vertices'], 1177)