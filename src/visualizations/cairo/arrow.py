from typing import Tuple

import cairo
import numpy as np

from visualizations.math.vector import Vec2


def draw_arrow(ctx: cairo.Context,
               start: Vec2,
               end: Vec2,
               head_size: float = 10,
               arrow_angle: float = np.pi / 2,
               color: Tuple[float, float, float] = (0, 0, 0),
               line_width: float = 1):
    # TODO: not quite correct yet. Arrow does not look too nice...

    direction_vector = -(end - start).normalize()
    direction_left = direction_vector.copy().rotate(-arrow_angle / 2)
    direction_right = direction_vector.copy().rotate(arrow_angle / 2)
    point_left = end + direction_left * head_size
    point_right = end + direction_right * head_size

    ctx.move_to(start.x, start.y)
    ctx.line_to(end.x, end.y)
    ctx.line_to(point_left.x, point_left.y)
    ctx.move_to(end.x, end.y)
    ctx.line_to(point_right.x, point_right.y)

    ctx.set_source_rgb(color[0], color[1], color[2])
    ctx.set_line_width(line_width)
    ctx.stroke()
