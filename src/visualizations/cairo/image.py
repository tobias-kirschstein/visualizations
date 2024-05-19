from typing import Optional

import cairo
import numpy as np

from visualizations.math.vector import Vec2


def _pre_translate(ctx, tx, ty):
    """Translate a cairo context without taking into account its
    scale and rotation"""
    mat = ctx.get_matrix()
    ctx.set_matrix(cairo.Matrix(mat[0], mat[1],
                                mat[2], mat[3],
                                mat[4] + tx, mat[5] + ty))


def draw_image(ctx: cairo.Context,
               image: np.ndarray,
               center: Vec2,
               size: Optional[Vec2] = None,
               angle: float = 0):
    """Draw a scaled image on a given context."""

    image_height, image_width, n_channels = image.shape
    if n_channels == 3:
        format = cairo.Format.RGB24
        # pycairo only understands 32bit pixel types...
        # Therefore, we need to artificially cripple the 24bit RGB image into a 32bit RGBA image where the
        # A channel is unused
        fill_alpha = np.ones((image_height, image_width, 1), dtype=np.uint8) * 255
        image = np.concatenate([image, fill_alpha], axis=2)
    else:
        format = cairo.Format.ARGB32
        # RGBA -> ARGB
        # image = np.concatenate([image[:, :, [3]], image[:, :, :3]], axis=2)
        image = np.array(image)

    image = np.ascontiguousarray(image)  # image needs to be contiguous

    # TODO: Ensure that transparent pixels have all 0s. Otherwise pycairo fucks up alpha blending
    image_surface = cairo.ImageSurface.create_for_data(
        image, format, image_width, image_height)

    if size is None:
        width = image.shape[1]
        height = image.shape[0]
    else:
        width = size.x
        height = size.y

    # calculate proportional scaling
    img_height, img_width = (image_surface.get_height(),
                             image_surface.get_width())
    scale_xy = min(1.0 * width / img_width, 1.0 * height / img_height)

    # scale, translate, and rotate the image around its center.
    ctx.save()
    ctx.rotate(angle)
    ctx.translate(-img_width / 2 * scale_xy, -img_height / 2 * scale_xy)

    if size is not None:
        ctx.scale(scale_xy, scale_xy)
    _pre_translate(ctx, center.x, center.y)
    ctx.set_source_surface(image_surface)

    ctx.paint()
    ctx.restore()


def to_image(surface: cairo.ImageSurface) -> np.ndarray:
    buf = surface.get_data()

    image = np.ndarray(shape=(surface.get_height(), surface.get_width(), 4),
                       dtype=np.uint8,
                       buffer=buf)

    format = surface.get_format()
    if format == cairo.Format.ARGB32:
        # ARGB -> RGBA
        pass
        # image = np.concatenate([image[:, :, 1:4], image[:, :, [0]]], axis=2)
    elif format == cairo.Format.RGB24:
        # Just crop last 8 bits. cairo uses them for speed reasons, but they are basically just padding
        image = image[:, :, :3]
    else:
        raise ValueError(f"Unsupported cairo image format {format}")

    return image
