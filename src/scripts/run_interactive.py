"""Interactive mode entry point (Phase 1).

Visual validation: render car + a non-self-intersecting curvy track.
Run with: python -m src.scripts.run_interactive
"""

import sys
import math

import pygame

from src.render.pygame_renderer import PygameRenderer
from src.sim.state import CarPose
from src.sim.track import Track


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Autonomous Driving 2D"
FPS = 60
BACKGROUND_COLOR = (20, 20, 20)
ROTATION_SPEED = 1.5  # rad/s


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

    pose = CarPose(x=WINDOW_WIDTH / 2, y=WINDOW_HEIGHT / 2, angle=0.0)

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
    

    try:
        running = True
        while running:
            dt = renderer.clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                pose.angle -= ROTATION_SPEED * dt
            if keys[pygame.K_RIGHT]:
                pose.angle += ROTATION_SPEED * dt

            renderer.screen.fill(BACKGROUND_COLOR)
            renderer.draw_track(renderer.screen, track)
            renderer.draw_car(renderer.screen, pose)
            pygame.display.flip()

    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
