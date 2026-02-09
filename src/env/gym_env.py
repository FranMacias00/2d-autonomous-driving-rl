"""Gymnasium environment wrapper for the 2D driving simulator."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.render.pygame_renderer import PygameRenderer
from src.sim.car import Car
from src.sim.sensors import SensorSuite
from src.sim.state import CarPose
from src.sim.track import Track


class DrivingEnv(gym.Env):
    """Gymnasium-compatible environment for the procedural driving task."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        self.render_mode = render_mode
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.car = Car(0.0, 0.0, 0.0)
        self.track: Optional[Track] = None
        self.sensors = SensorSuite()
        self.renderer: Optional[PygameRenderer] = None
        self.steps = 0

    def _make_centerline(self) -> list[tuple[float, float]]:
        n_points = 80
        x0, x1 = 80.0, 720.0
        y_mid = 320.0
        margin = 80.0
        height = 600.0

        amplitude = float(self.np_random.uniform(70.0, 130.0))
        waves = float(self.np_random.uniform(0.6, 1.2))

        points: list[tuple[float, float]] = []
        for idx in range(n_points):
            t = idx / (n_points - 1)
            x = x0 + (x1 - x0) * t
            y = y_mid + amplitude * math.sin(2.0 * math.pi * waves * t)
            y = max(margin, min(height - margin, y))
            points.append((x, y))
        return points

    def _spawn_forward(self, track: Track) -> Tuple[float, float, float]:
        start = track.centerline[0]
        next_point = track.centerline[1]
        angle = math.atan2(next_point[1] - start[1], next_point[0] - start[0])
        spawn_x = start[0] + math.cos(angle) * 60.0
        spawn_y = start[1] + math.sin(angle) * 60.0
        return spawn_x, spawn_y, angle

    def _get_observation(self) -> np.ndarray:
        rays = self.sensors.cast(self.car, self.track)
        distances = [ray["distance"] / self.sensors.max_range for ray in rays]
        velocity_norm = self.car.velocity / self.car.max_speed
        velocity_norm = float(np.clip(velocity_norm, 0.0, 1.0))
        obs = np.array(distances + [velocity_norm], dtype=np.float32)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.track = Track(centerline=self._make_centerline(), road_width=110.0)
        spawn_x, spawn_y, angle = self._spawn_forward(self.track)
        self.car.reset(spawn_x, spawn_y, angle)

        observation = self._get_observation()
        info: dict = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))

        self.car.throttle = throttle
        self.car.steering = steering

        front_old = self.car.front_point()
        self.car.step(1.0 / 60.0)
        front_new = self.car.front_point()

        off_track = not self.track.is_car_on_road(self.car)
        finish = self.track.has_crossed_finish(front_old, front_new)

        reward = 0.01 + 0.1 * (self.car.velocity / self.car.max_speed)
        if off_track:
            reward -= 20.0
        if finish:
            reward += 100.0

        self.steps += 1
        terminated = off_track or finish
        truncated = self.steps >= 1500

        event = None
        if finish:
            event = "finish"
        elif off_track:
            event = "off_track"
        elif truncated:
            event = "timeout"

        info = {"event": event}
        observation = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        if self.renderer is None:
            self.renderer = PygameRenderer(width=800, height=600)

        self.renderer.screen.fill((20, 20, 20))
        if self.track is not None:
            self.renderer.draw_track(self.renderer.screen, self.track)

        pose = CarPose(x=self.car.x, y=self.car.y, angle=self.car.angle)
        self.renderer.draw_car(self.renderer.screen, pose)
        rays = self.sensors.cast(self.car, self.track)
        self.renderer.draw_sensors(
            self.renderer.screen, rays, danger_distance=self.sensors.danger_distance
        )
        self.renderer.draw_text(
            self.renderer.screen,
            f"Velocity: {self.car.velocity:.1f} px/s",
            (20, 20),
        )

        import pygame

        pygame.display.flip()
        self.renderer.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self.renderer is None:
            return
        import pygame

        pygame.quit()
        self.renderer = None
