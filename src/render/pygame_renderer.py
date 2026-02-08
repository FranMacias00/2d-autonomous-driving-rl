"""Pygame rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Tuple

import pygame

from src.sim.state import CarPose
from src.sim.track import Track


Point = Tuple[float, float]


@dataclass
class CarGeometry:
    """Geometry dimensions for the top-down car drawing."""

    length: float = 80.0
    width: float = 40.0
    cabin_length: float = 40.0
    cabin_width: float = 26.0
    wheel_length: float = 16.0
    wheel_width: float = 8.0
    wheel_offset_x: float = 28.0
    wheel_offset_y: float = 18.0


class PygameRenderer:
    """Renderer for drawing a simple top-down car using Pygame primitives."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.geometry = CarGeometry()
        self.font = pygame.font.SysFont("arial", 18)

    @staticmethod
    def _rotate_point(point: Point, angle: float) -> Point:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x, y = point
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    @classmethod
    def _transform_points(cls, points: Iterable[Point], pose: CarPose) -> List[Point]:
        transformed: List[Point] = []
        for point in points:
            rotated = cls._rotate_point(point, pose.angle)
            transformed.append((rotated[0] + pose.x, rotated[1] + pose.y))
        return transformed

    @staticmethod
    def _rect_points(center: Point, length: float, width: float) -> List[Point]:
        cx, cy = center
        half_l = length / 2.0
        half_w = width / 2.0
        return [
            (cx - half_l, cy - half_w),
            (cx + half_l, cy - half_w),
            (cx + half_l, cy + half_w),
            (cx - half_l, cy + half_w),
        ]

    def draw_car(self, surface: pygame.Surface, pose: CarPose) -> None:
        geometry = self.geometry

        chassis = self._rect_points((0.0, 0.0), geometry.length, geometry.width)
        cabin_center = (geometry.length * 0.1, 0.0)
        cabin = self._rect_points(cabin_center, geometry.cabin_length, geometry.cabin_width)

        wheel_offsets = [
            (geometry.wheel_offset_x, geometry.wheel_offset_y),
            (geometry.wheel_offset_x, -geometry.wheel_offset_y),
            (-geometry.wheel_offset_x, geometry.wheel_offset_y),
            (-geometry.wheel_offset_x, -geometry.wheel_offset_y),
        ]
        wheels = [
            self._rect_points(offset, geometry.wheel_length, geometry.wheel_width)
            for offset in wheel_offsets
        ]

        chassis_points = self._transform_points(chassis, pose)
        cabin_points = self._transform_points(cabin, pose)
        wheel_points = [self._transform_points(wheel, pose) for wheel in wheels]

        pygame.draw.polygon(surface, (30, 130, 200), chassis_points)
        pygame.draw.polygon(surface, (80, 180, 240), cabin_points)
        for wheel in wheel_points:
            pygame.draw.polygon(surface, (10, 10, 10), wheel)

    def draw_track(self, surface: pygame.Surface, track: Track) -> None:
        left_border, right_border = track.get_borders()
        pygame.draw.lines(surface, (255, 255, 255), False, left_border, 3)
        pygame.draw.lines(surface, (255, 255, 255), False, right_border, 3)
        pygame.draw.line(surface, (220, 40, 40), left_border[-1], right_border[-1], 10)
        pygame.draw.lines(surface, (140, 140, 140), False, track.centerline, 1)

    def draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        position: Point,
        color: Tuple[int, int, int] = (240, 240, 240),
    ) -> None:
        text_surface = self.font.render(text, True, color)
        surface.blit(text_surface, position)

    def draw_sensors(self, surface: pygame.Surface, rays_data: List[dict]) -> None:
        for ray in rays_data:
            distance = ray["distance"]
            hit_point = ray["hit_point"]
            color = (40, 200, 40) if distance >= 50.0 or hit_point is None else (220, 40, 40)
            pygame.draw.line(surface, color, ray["start_pos"], ray["end_pos"], 2)
            if hit_point is not None:
                pygame.draw.circle(surface, color, hit_point, 3)
