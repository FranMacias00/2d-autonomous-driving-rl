"""Track geometry based on a centerline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple


Point = Tuple[float, float]


@dataclass
class Track:
    """Centerline-based track definition."""

    centerline: List[Point]
    road_width: float

    def get_borders(self) -> Tuple[List[Point], List[Point]]:
        """Compute left and right borders from the centerline."""

        if len(self.centerline) < 2:
            raise ValueError("Centerline must contain at least two points.")

        normals: List[Point] = []
        for start, end in zip(self.centerline[:-1], self.centerline[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            angle = math.atan2(dy, dx)
            normal_angle = angle + math.pi / 2.0
            normals.append((math.cos(normal_angle), math.sin(normal_angle)))

        left_border: List[Point] = []
        right_border: List[Point] = []
        half_width = self.road_width / 2.0

        for idx, point in enumerate(self.centerline):
            if idx == 0:
                normal = normals[0]
            elif idx == len(self.centerline) - 1:
                normal = normals[-1]
            else:
                normal = (
                    normals[idx - 1][0] + normals[idx][0],
                    normals[idx - 1][1] + normals[idx][1],
                )
                length = math.hypot(normal[0], normal[1])
                if length != 0:
                    normal = (normal[0] / length, normal[1] / length)

            left_border.append(
                (point[0] + normal[0] * half_width, point[1] + normal[1] * half_width)
            )
            right_border.append(
                (point[0] - normal[0] * half_width, point[1] - normal[1] * half_width)
            )

        return left_border, right_border

    def get_border_segments(self) -> List[Tuple[Point, Point]]:
        """Return combined segments for left and right borders."""

        left_border, right_border = self.get_borders()
        segments: List[Tuple[Point, Point]] = []
        for border in (left_border, right_border):
            segments.extend(list(zip(border[:-1], border[1:])))
        return segments
