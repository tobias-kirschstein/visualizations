from math import sqrt, ceil
from pathlib import Path

import matplotlib.pyplot as plt

from visualizations.render.single_mesh import render_single_mesh

NPHM_ALL_SCANS_PATH = "D:/Projects/NPHM/data/best_scans"
N_HEADS = 4

if __name__ == '__main__':
    plt.figure()

    nrows = ceil(sqrt(N_HEADS))

    for i_head, mesh_file in enumerate(Path(NPHM_ALL_SCANS_PATH).iterdir()):
        if i_head >= N_HEADS:
            break

        rendered_img = render_single_mesh(mesh_file, scale=1/10, crop_y_min=-17)
        plt.subplot(nrows, nrows, i_head + 1)
        plt.imshow(rendered_img)
        plt.axis('off')

    plt.tight_layout()

    plt.show()