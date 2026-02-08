"""Track geometry based on a centerline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.sim.car import Car


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

    def get_finish_segment(self) -> Tuple[Point, Point]:
        """Return the finish line segment using the last border points."""
        left_border, right_border = self.get_borders()
        return left_border[-1], right_border[-1]

    def has_crossed_finish(self, p0: Point, p1: Point) -> bool:
        """Check whether segment p0->p1 intersects the finish line."""
        finish_start, finish_end = self.get_finish_segment()

        def orientation(a: Point, b: Point, c: Point) -> int:
            val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            eps = 1e-9
            if abs(val) <= eps:
                return 0
            return 1 if val > 0 else 2

        def on_segment(a: Point, b: Point, c: Point) -> bool:
            return (
                min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
                and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
            )

        o1 = orientation(p0, p1, finish_start)
        o2 = orientation(p0, p1, finish_end)
        o3 = orientation(finish_start, finish_end, p0)
        o4 = orientation(finish_start, finish_end, p1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p0, finish_start, p1):
            return True
        if o2 == 0 and on_segment(p0, finish_end, p1):
            return True
        if o3 == 0 and on_segment(finish_start, p0, finish_end):
            return True
        if o4 == 0 and on_segment(finish_start, p1, finish_end):
            return True

        return False

    def is_point_on_road(self, point: Point) -> bool:
        """Check whether a point is strictly inside the road polygon."""
        left_border, right_border = self.get_borders()
        polygon = left_border + list(reversed(right_border))

        x, y = point
        epsilon = 1e-9

        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            (x1, y1), (x2, y2) = start, end
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if abs(cross) <= epsilon:
                dot = (x - x1) * (x - x2) + (y - y1) * (y - y2)
                if dot <= epsilon:
                    return False

        crossings = 0
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            x1, y1 = start
            x2, y2 = end
            if (y1 > y) != (y2 > y):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x_intersect > x:
                    crossings += 1
        return crossings % 2 == 1

    def is_car_on_road(self, car: "Car") -> bool:
        """Check whether the entire car bounding box is within the road."""
        return all(self.is_point_on_road(vertex) for vertex in car.get_bbox_vertices())
