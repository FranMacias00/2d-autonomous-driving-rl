"""Ray-casting sensors for the vehicle."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

from src.sim.track import Track


Point = Tuple[float, float]
Segment = Tuple[Point, Point]


@dataclass
class SensorSuite:
    """Cast multiple rays around the car to detect nearby borders."""

    num_rays: int = 9 # Subimos de 7 a 9 para no perder resolución
    fov_degrees: float = 180.0 # Ampliamos de 120 a 180 para ver los laterales
    max_range: float = 220.0
    danger_distance: float = 50.0
    front_offset: float = 40.0

    def _ray_directions(self, angle: float) -> List[Point]:
        if self.num_rays <= 1:
            return [(math.cos(angle), math.sin(angle))]

        half_fov = math.radians(self.fov_degrees) / 2.0
        step = (2.0 * half_fov) / (self.num_rays - 1)
        directions: List[Point] = []
        for idx in range(self.num_rays):
            ray_angle = angle - half_fov + step * idx
            directions.append((math.cos(ray_angle), math.sin(ray_angle)))
        return directions

    @staticmethod
    def _cross(a: Point, b: Point) -> float:
        return a[0] * b[1] - a[1] * b[0]

    def _ray_segment_intersection(
        self, origin: Point, direction: Point, segment: Segment
    ) -> Optional[Tuple[float, Point]]:
        p = origin
        r = direction
        q = segment[0]
        s = (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
        denominator = self._cross(r, s)
        if abs(denominator) < 1e-9:
            return None

        qp = (q[0] - p[0], q[1] - p[1])
        t = self._cross(qp, s) / denominator
        u = self._cross(qp, r) / denominator
        if t > self.max_range:
            return None
        if t >= 0.0 and 0.0 <= u <= 1.0:
            hit_point = (p[0] + t * r[0], p[1] + t * r[1])
            return t, hit_point
        return None

    def cast(self, car: object, track: Track) -> List[dict]:
        origin = (
            car.x + math.cos(car.angle) * self.front_offset,
            car.y + math.sin(car.angle) * self.front_offset,
        )
        segments = track.get_border_segments()
        directions = self._ray_directions(car.angle)
        rays_data: List[dict] = []

        for direction in directions:
            closest_distance = self.max_range
            closest_point: Optional[Point] = None

            for segment in segments:
                hit = self._ray_segment_intersection(origin, direction, segment)
                if hit is None:
                    continue
                distance, hit_point = hit
                if distance < closest_distance:
                    closest_distance = distance
                    closest_point = hit_point

            if closest_point is None:
                end_pos = (
                    origin[0] + direction[0] * self.max_range,
                    origin[1] + direction[1] * self.max_range,
                )
                distance = self.max_range
            else:
                end_pos = closest_point
                distance = closest_distance

            rays_data.append(
                {
                    "start_pos": origin,
                    "end_pos": end_pos,
                    "distance": distance,
                    "hit_point": closest_point,
                }
            )

        return rays_data
