"""Interactive mode entry point (Phase 1).

Visual validation: render car + a non-self-intersecting curvy track.
Run with: python -m src.scripts.run_interactive
"""

import math
import sys

import pygame

from src.render.pygame_renderer import PygameRenderer
from src.sim.car import Car
from src.sim.sensors import SensorSuite
from src.sim.state import CarPose
from src.sim.track import Track


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Autonomous Driving 2D"
FPS = 60
BACKGROUND_COLOR = (20, 20, 20)


def make_centerline(
    n: int = 55,
    x0: float = 80.0,
    x1: float = 720.0,
    y_mid: float = 320.0,
    amplitude: float = 170.0,
    waves: float = 1.8,
    margin: float = 60.0,
) -> list[tuple[float, float]]:
    """Generate a smooth, non-self-intersecting centerline.

    Uses monotonic x progression + sinusoidal y oscillation.
    """
    pts: list[tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + (x1 - x0) * t
        y = y_mid + amplitude * math.sin(2.0 * math.pi * waves * t)
        y = max(margin, min(WINDOW_HEIGHT - margin, y))  # keep inside window
        pts.append((x, y))
    return pts


def main() -> int:
    pygame.init()
    renderer = PygameRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    pygame.display.set_caption(WINDOW_TITLE)

    # Curvy, larger track without overlaps
    track = Track(
        centerline=make_centerline(
            n=80,
            amplitude=110.0,  # antes 170
            waves=0.9,        # antes 1.8  ← ESTE ES EL CLAVE
            margin=80.0,
        ),
        road_width=110.0,     # antes 140
    )

    start_x, start_y = track.centerline[0]
    next_x, next_y = track.centerline[1]
    start_angle = math.atan2(next_y - start_y, next_x - start_x)
    car = Car(x=start_x, y=start_y, angle=start_angle)
    spawn_pose = (start_x, start_y, start_angle)
    sensors = SensorSuite()

    try:
        running = True
        while running:
            dt = renderer.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    car.reset(*spawn_pose)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                car.throttle = 1.0
            elif keys[pygame.K_s]:
                car.throttle = -1.0
            else:
                car.throttle = 0.0

            if keys[pygame.K_a]:
                car.steering = -1.0
            elif keys[pygame.K_d]:
                car.steering = 1.0
            else:
                if car.steering > 0.0:
                    car.steering = max(0.0, car.steering - car.steering_return_rate * dt)
                elif car.steering < 0.0:
                    car.steering = min(0.0, car.steering + car.steering_return_rate * dt)

            car.step(dt)
            on_road = track.is_car_on_road(car)

            renderer.screen.fill(BACKGROUND_COLOR)
            renderer.draw_track(renderer.screen, track)

            pose = CarPose(x=car.x, y=car.y, angle=car.angle)
            renderer.draw_car(renderer.screen, pose)
            rays_data = sensors.cast(car, track)
            renderer.draw_sensors(renderer.screen, rays_data, danger_distance=sensors.danger_distance)

            speed_kmh = car.velocity * 3.6
            velocity_text = f"Velocity: {car.velocity:.1f} px/s ({speed_kmh:.1f} km/h)"
            angle_text = f"Angle: {car.angle:.2f} rad"
            renderer.draw_text(renderer.screen, velocity_text, (20, 20))
            renderer.draw_text(renderer.screen, angle_text, (20, 45))
            if not on_road:
                off_track_text = "OFF TRACK"
                text_surface = renderer.font.render(off_track_text, True, (220, 40, 40))
                text_rect = text_surface.get_rect(center=(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2))
                renderer.screen.blit(text_surface, text_rect)
            pygame.display.flip()

    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
