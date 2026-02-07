"""Interactive mode entry point (Phase 1).

Minimal Pygame window with a stable game loop.
"""

import sys

import pygame


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Autonomous Driving 2D"
FPS = 60
BACKGROUND_COLOR = (20, 20, 20)
TEXT_COLOR = (230, 230, 230)
MESSAGE = "Interactive mode (Phase 1)"


def main() -> int:
    pygame.init()
    try:
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)

        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        text_surface = font.render(MESSAGE, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            screen.fill(BACKGROUND_COLOR)
            screen.blit(text_surface, text_rect)
            pygame.display.flip()
            clock.tick(FPS)
    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
