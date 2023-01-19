import matplotlib.pyplot as plt
import numpy as np
import trimesh
from elias.util.io import save_img

from visualizations.env_nphm import NPHM_DATA_PATH
import pyvista as pv
from scipy.spatial.transform.rotation import Rotation as R

from visualizations.render.single_mesh import render_single_mesh

POINTCLOUD_PATH = f"{NPHM_DATA_PATH}/results_identity/PC/s_barbara_re_e_10.npy"
FITTED_NPHM_PATH = f"{NPHM_DATA_PATH}/results_identity/NPHM/s_barbara_re_e_10.ply"

RESOLUTION = 2048
IMAGE_WIDTH_POINTCLOUD = int(3 / 4 * RESOLUTION)
IMAGE_HEIGHT_POINTCLOUD = RESOLUTION
ROTATION = 0.3 * np.pi


def scaled_cmap(cmap_name: str, scale: float):
    cmap = plt.get_cmap(cmap_name)
    return lambda x: cmap(x * scale)


if __name__ == '__main__':
    # Plot pointcloud
    pointcloud = np.load(POINTCLOUD_PATH)

    p = pv.Plotter(off_screen=True)
    p.window_size = (IMAGE_WIDTH_POINTCLOUD, IMAGE_HEIGHT_POINTCLOUD)
    p.camera_set = True
    p.camera_position = 'xy'
    p.camera.position = (0, 0, 3)  # Move camera a bit left for pointcloud

    # rotation_matrix = R.from_euler('y', -105 + 50, degrees=True).as_matrix()
    rotation_matrix_1 = R.from_euler(seq='xyz', angles=[0, -50 / 360 * 2 * np.pi, 0]).as_matrix()
    rotation_matrix_2 = R.from_euler(seq='xyz', angles=[0, ROTATION, 0]).as_matrix()
    pointcloud_canonical = pointcloud @ rotation_matrix_1.T
    pointcloud = pointcloud_canonical @ rotation_matrix_2.T
    p.theme.render_points_as_spheres = True
    p.add_points(pointcloud, scalars=-pointcloud_canonical[:, 2], cmap=scaled_cmap('turbo', 1.3), point_size=10)

    p.remove_scalar_bar()
    rendered_mesh = p.screenshot(transparent_background=True)

    save_img(rendered_mesh, f"{NPHM_DATA_PATH}/teaser/pointcloud_fitting.png")

    exit(1)
    # Render head
    mesh = trimesh.load(FITTED_NPHM_PATH)
    angle = ROTATION
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    mesh.apply_transform(rotation_matrix)
    rendered_mesh = render_single_mesh(mesh, use_orthographic_cam=False, fov=np.pi / 6,
                                       image_height=IMAGE_HEIGHT_POINTCLOUD, image_width=IMAGE_WIDTH_POINTCLOUD)
    save_img(rendered_mesh, f"{NPHM_DATA_PATH}/teaser/nphm_fitting.png")
