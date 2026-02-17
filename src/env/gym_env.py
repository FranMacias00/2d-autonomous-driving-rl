"""Gymnasium environment wrapper for the 2D driving simulator."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.render.pygame_renderer import PygameRenderer
from src.sim.car import Car
from src.sim.sensors import SensorSuite
from src.sim.state import CarPose
from src.sim.track import Track


class DrivingEnv(gym.Env):
    """Gymnasium-compatible environment for the procedural driving task."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        
        # Espacio de acciones: [throttle, steering]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Espacio de observaciones: 7 sensores + 1 velocidad
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.car = Car(0.0, 0.0, 0.0)
        self.track: Optional[Track] = None
        self.sensors = SensorSuite()
        self.renderer: Optional[PygameRenderer] = None
        self.steps = 0

    def _make_centerline(self) -> list[tuple[float, float]]:
        """Genera una pista aleatoria usando el generador de números de Gym."""
        n_points = 80
        x0, x1 = 80.0, 720.0
        y_mid = 320.0
        margin = 80.0
        height = 600.0

        # Usamos self.np_random para asegurar la reproducibilidad con 'seed'
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
        """Calcula la posición inicial encarando la pista."""
        start = track.centerline[0]
        next_point = track.centerline[1]
        angle = math.atan2(next_point[1] - start[1], next_point[0] - start[0])
        spawn_x = start[0] + math.cos(angle) * 60.0
        spawn_y = start[1] + math.sin(angle) * 60.0
        return spawn_x, spawn_y, angle

    def _get_observation(self) -> np.ndarray:
        """Obtiene y normaliza los datos de los sensores y velocidad."""
        rays = self.sensors.cast(self.car, self.track)
        # Normalizamos distancias al rango [0, 1]
        distances = [np.clip(ray["distance"] / self.sensors.max_range, 0.0, 1.0) for ray in rays]
        
        # Eliminamos el ABS: ahora el coche distingue si la velocidad es negativa (marcha atrás)
        # Escalamos de [-max_speed, max_speed] a [0, 1] donde 0.5 es parado
        velocity_norm = np.clip((self.car.velocity / self.car.max_speed + 1.0) / 2.0, 0.0, 1.0)
        
        return np.array(distances + [velocity_norm], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reinicia el entorno con una pista nueva."""
        super().reset(seed=seed)
        self.steps = 0
        
        # Crear pista nueva con parámetros aleatorios
        self.track = Track(centerline=self._make_centerline(), road_width=110.0)
        
        # Posicionar coche
        spawn_x, spawn_y, angle = self._spawn_forward(self.track)
        self.car.reset(spawn_x, spawn_y, angle)

        observation = self._get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """Ejecuta un paso de simulación."""
        # Limpiar y asegurar formato de acción
        action = np.clip(action, -1.0, 1.0)
        self.car.throttle = float(action[0])
        self.car.steering = float(action[1])

        # Guardar posición anterior para detectar cruce de meta
        front_old = self.car.front_point()
        self.car.step(1.0 / 60.0)
        front_new = self.car.front_point()

        # Detección de estado
        off_track = not self.track.is_car_on_road(self.car)
        finish = self.track.has_crossed_finish(front_old, front_new)

        # Cálculo de distancia a meta para margen de 50px
        f_start, f_end = self.track.get_finish_segment()
        m_center = ((f_start[0] + f_end[0]) / 2, (f_start[1] + f_end[1]) / 2)
        dist_to_finish = math.hypot(self.car.x - m_center[0], self.car.y - m_center[1])

        reward = 0.0
        terminated = False
        event = None

        # --- LÓGICA DE RECOMPENSAS Y ZONA DE GRACIA ---
        if finish:
            reward = 150.0
            terminated = True
            event = "finish"
        
        # Muerte: Fuera de pista y LEJOS de la meta (> 50px)
        elif off_track and dist_to_finish > self.car.length:
            reward = -30.0
            terminated = True
            event = "off_track"
            
        # Zona de Gracia: Fuera de pista pero CERCA de la meta (<= 50px)
        elif off_track and dist_to_finish <= self.car.length:
            # Recompensa plana para incentivar el cierre sin premiar velocidad en césped
            reward = 0.05 
        
        # Conducción Normal (en pista)
        else:
            if self.car.velocity < 1.0:
                reward = -0.1
            else:
                reward = (0.2 * (self.car.velocity / self.car.max_speed)) + 0.05
        
        self.steps += 1
        truncated = self.steps >= 1500
        if truncated and not terminated:
            event = "timeout"

        # --- BLOQUE DE PRINTS DE DIAGNÓSTICO ---
        if terminated or truncated:
            print(f"\n--- DEBUG EPISODIO FINALIZADO ---")
            print(f"¿Detección off_track?: {off_track}")
            print(f"¿Detección finish?: {finish}")
            print(f"Evento asignado en info: {event}")
            print(f"Recompensa de este último paso: {reward:.2f}")
            print(f"Velocidad actual: {self.car.velocity:.2f}")
            print(f"Posición morro (front_new): {front_new}")
            print(f"----------------------------------")

        info = {"event": event}
        observation = self._get_observation()

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Dibuja el entorno si el modo es 'human'."""
        if self.render_mode != "human":
            return
            
        if self.renderer is None:
            self.renderer = PygameRenderer(width=800, height=600)

        # Proceso de dibujo
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
            f"Velocidad: {self.car.velocity:.1f} px/s | Pasos: {self.steps}",
            (20, 20),
        )

        pygame.display.flip()
        # Limitador de FPS: solo frena la CPU cuando estamos mirando
        self.renderer.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Cierra el renderer y libera recursos."""
        if self.renderer:
            pygame.quit()
            self.renderer = None