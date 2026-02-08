"""Interactive mode entry point (Phase 1).

Minimal Pygame window with a stable game loop.
"""

import sys

import pygame

from src.render.pygame_renderer import PygameRenderer
from src.sim.state import CarPose
from src.sim.track import Track


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Autonomous Driving 2D"
FPS = 60
BACKGROUND_COLOR = (20, 20, 20)
ROTATION_SPEED = 1.5


def main() -> int:
    pygame.init()
    renderer = PygameRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    pygame.display.set_caption(WINDOW_TITLE)
    pose = CarPose(x=WINDOW_WIDTH / 2, y=WINDOW_HEIGHT / 2, angle=0.0)
    track = Track(
        centerline=[
            (120.0, 420.0),
            (260.0, 360.0),
            (420.0, 320.0),
            (580.0, 260.0),
            (700.0, 180.0),
        ],
        road_width=140.0,
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
