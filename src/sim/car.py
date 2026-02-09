"""Car dynamics model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Car:
    """Simple kinematic car with throttle and steering inputs."""

    x: float
    y: float
    angle: float
    velocity: float = 0.0
    throttle: float = 0.0
    steering: float = 0.0
    max_speed: float = 200.0
    max_accel: float = 150.0
    drag: float = 1.8
    steering_rate: float = 3.0
    steering_return_rate: float = 6.0
    length: float = 80.0
    width: float = 40.0

    def step(self, dt: float) -> None:
        """Advance the car state by dt seconds."""
        accel = self.throttle * self.max_accel
        self.velocity += accel * dt
        self.velocity -= self.drag * self.velocity * dt

        max_reverse = -self.max_speed / 2.0
        if self.velocity > self.max_speed:
            self.velocity = self.max_speed
        elif self.velocity < max_reverse:
            self.velocity = max_reverse

        if abs(self.velocity) > 1e-3:
            steer_scale = abs(self.velocity) / self.max_speed
            self.angle += self.steering * self.steering_rate * steer_scale * dt

        self.x += math.cos(self.angle) * self.velocity * dt
        self.y += math.sin(self.angle) * self.velocity * dt

    def get_bbox_vertices(self) -> List[Tuple[float, float]]:
        """Return the global coordinates for the four car corners."""
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        local_corners = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]

        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        corners: List[Tuple[float, float]] = []
        for local_x, local_y in local_corners:
            rotated_x = local_x * cos_a - local_y * sin_a
            rotated_y = local_x * sin_a + local_y * cos_a
            corners.append((self.x + rotated_x, self.y + rotated_y))
        return corners

    def front_point(self) -> Tuple[float, float]:
        """Return the point at the front center of the car."""
        half_length = self.length / 2.0
        return (
            self.x + math.cos(self.angle) * half_length,
            self.y + math.sin(self.angle) * half_length,
        )

    def reset(self, x: float, y: float, angle: float) -> None:
        """Reset car pose and clear motion inputs."""
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0.0
        self.throttle = 0.0
        self.steering = 0.0
