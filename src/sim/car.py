"""Car dynamics model."""

from __future__ import annotations

import math
from dataclasses import dataclass


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
