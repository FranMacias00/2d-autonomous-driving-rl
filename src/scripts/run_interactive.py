"""Interactive mode - Procedural Learning Environment.

Cada episodio genera un circuito único para obligar a la IA (o al humano)
a generalizar y no memorizar la pista.
"""

import math
import sys
import random
import pygame

from src.render.pygame_renderer import PygameRenderer
from src.sim.car import Car
from src.sim.sensors import SensorSuite
from src.sim.state import CarPose
from src.sim.track import Track

# Configuración de ventana
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Autonomous Driving 2D - Procedural Generation"
FPS = 60
BACKGROUND_COLOR = (20, 20, 20)

def make_random_centerline() -> list[tuple[float, float]]:
    """Genera una línea central con parámetros aleatorios para cada episodio."""
    # Variamos la dificultad: más ondas y más amplitud = más difícil
    n = 80
    x0, x1 = 80.0, 720.0
    y_mid = 320.0
    
    # Parámetros aleatorios
    amplitude = random.uniform(70.0, 130.0)
    waves = random.uniform(0.6, 1.2)
    margin = 80.0
    
    pts: list[tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + (x1 - x0) * t
        y = y_mid + amplitude * math.sin(2.0 * math.pi * waves * t)
        y = max(margin, min(WINDOW_HEIGHT - margin, y))
        pts.append((x, y))
    return pts

def reset_scenario(car: Car):
    """Crea una nueva pista y reposiciona el coche."""
    # 1. Generar nuevo Track
    new_track = Track(
        centerline=make_random_centerline(),
        road_width=110.0
    )
    
    # 2. Calcular nueva posición de spawn basada en la nueva pista
    s_pts = new_track.centerline
    s_angle = math.atan2(s_pts[1][1] - s_pts[0][1], s_pts[1][0] - s_pts[0][0])
    
    spawn_x = s_pts[0][0] + math.cos(s_angle) * 60
    spawn_y = s_pts[0][1] + math.sin(s_angle) * 60
    
    # 3. Resetear coche
    car.reset(spawn_x, spawn_y, s_angle)
    
    return new_track

def main() -> int:
    pygame.init()
    renderer = PygameRenderer(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    pygame.display.set_caption(WINDOW_TITLE)

    # Inicialización del primer episodio
    # Creamos un coche "dummy" que el reset_scenario configurará
    car = Car(0, 0, 0)
    track = reset_scenario(car)
    
    sensors = SensorSuite()
    finish_reached = False
    episode_count = 1

    try:
        running = True
        while running:
            dt = renderer.clock.tick(FPS) / 1000.0

            # 1. EVENTOS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        # R manual también genera circuito nuevo
                        track = reset_scenario(car)
                        finish_reached = False
                        episode_count += 1

            # 2. LÓGICA DE CONDUCCIÓN
            if not finish_reached:
                keys = pygame.key.get_pressed()
                car.throttle = 1.0 if keys[pygame.K_w] else -1.0 if keys[pygame.K_s] else 0.0
                
                if keys[pygame.K_a]:
                    car.steering = -1.0
                elif keys[pygame.K_d]:
                    car.steering = 1.0
                else:
                    if car.steering > 0.0:
                        car.steering = max(0.0, car.steering - car.steering_return_rate * dt)
                    elif car.steering < 0.0:
                        car.steering = min(0.0, car.steering + car.steering_return_rate * dt)

                # Movimiento
                front_old = car.front_point()
                car.step(dt)
                front_new = car.front_point()

                # --- LÓGICA DE EPISODIOS (Reset en Meta o Choque) ---
                
                # A. ¿Cruzó la meta?
                if track.has_crossed_finish(front_old, front_new):
                    print(f"Episodio {episode_count} COMPLETADO. Generando nuevo...")
                    # En modo interactivo podrías querer ver el FINISH un momento,
                    # pero para RL el reset es instantáneo. Aquí lo hacemos manual con 'R'
                    # o automático si prefieres. Vamos a dejarlo marcado como FINISH.
                    finish_reached = True

                # B. ¿Se salió de la pista? (HARD RESET para PPO)
                else:
                    on_road = track.is_car_on_road(car)
                    # Usamos distancia dinámica a meta para no resetear por error al ganar
                    f_start, f_end = track.get_finish_segment()
                    m_center = ((f_start[0] + f_end[0]) / 2, (f_start[1] + f_end[1]) / 2)
                    dist_to_finish = math.hypot(car.x - m_center[0], car.y - m_center[1])

                    if not on_road and dist_to_finish > car.length:
                        print(f"COLISIÓN Episodio {episode_count}. Reiniciando...")
                        track = reset_scenario(car)
                        episode_count += 1

            # 3. RENDERIZADO
            renderer.screen.fill(BACKGROUND_COLOR)
            renderer.draw_track(renderer.screen, track)
            
            pose = CarPose(x=car.x, y=car.y, angle=car.angle)
            renderer.draw_car(renderer.screen, pose)
            
            # Sensores (indispensables para que la IA "vea" la nueva pista)
            rays_data = sensors.cast(car, track)
            renderer.draw_sensors(renderer.screen, rays_data, danger_distance=sensors.danger_distance)

            # UI Información
            renderer.draw_text(renderer.screen, f"Episode: {episode_count}", (20, 20))
            renderer.draw_text(renderer.screen, f"Velocity: {car.velocity:.1f} px/s", (20, 45))

            if finish_reached:
                # Texto de meta (Pulsa R para el siguiente circuito aleatorio)
                finish_text = "FINISH! Press 'R' for next track"
                text_surf = renderer.font.render(finish_text, True, (40, 200, 80))
                text_rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
                renderer.screen.blit(text_surf, text_rect)

            pygame.display.flip()

    finally:
        pygame.quit()

    return 0

if __name__ == "__main__":
    sys.exit(main())