import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    identities = range(8)
    expressions = range(5)

    nrows, ncols = len(identities), len(expressions)

    rendered_head = np.zeros((1024, 512))
    rendered_head[287:812, 100: 400] = 1

    plt.figure(figsize=(10, 16))
    for i_identiy, identity in enumerate(identities):
        for i_expression, expression in enumerate(expressions):
            i_image = i_identiy * ncols + i_expression

            plt.subplot(nrows, ncols, i_image + 1)
            plt.imshow(rendered_head)
            plt.axis('off')

    # TODO: How to draw a line over everything? Maybe need a
    #   different plotting library?
    plt.axvline(1, ymin=0, ymax=100000)
    plt.tight_layout()
    plt.show()

