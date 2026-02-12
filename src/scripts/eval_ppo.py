"""Evaluación visual de un agente PPO entrenado en DrivingEnv."""

from __future__ import annotations

import os

from stable_baselines3 import PPO

from src.env.gym_env import DrivingEnv


MODEL_PATH = "models/ppo_driving_car_v2_20260212_112538.zip"
N_EPISODES = 3


def _get_end_reason(terminated: bool, truncated: bool, info: dict) -> str:
    """Normaliza el motivo de fin al formato solicitado."""
    event = info.get("event")

    if event == "finish":
        return "finish"
    if event == "off_track":
        return "off_track"
    if event in {"timeout", "time_limit"} or truncated:
        return "time_limit"
    if terminated:
        return "terminated"
    return "unknown"


def main() -> None:
    """Carga un PPO entrenado y lo evalúa visualmente en 3 episodios."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{MODEL_PATH}'. "
            "Entrena primero o verifica la ruta."
        )

    # Entorno visual: render_mode='human' activa pygame en tiempo real.
    # DrivingEnv ya limita el render a 60 FPS internamente con clock.tick(...).
    env = DrivingEnv(render_mode="human")

    # Cargar el modelo entrenado sin reentrenar.
    model = PPO.load(MODEL_PATH, env=env)

    try:
        for episode in range(1, N_EPISODES + 1):
            obs, _ = env.reset()
            steps = 0
            total_reward = 0.0

            while True:
                # deterministic=True para una política estable durante evaluación.
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                steps += 1
                total_reward += reward

                if terminated or truncated:
                    end_reason = _get_end_reason(terminated, truncated, info)
                    print(
                        f"Episodio {episode} | "
                        f"pasos={steps} | "
                        f"reward_acumulado={total_reward:.2f} | "
                        f"fin={end_reason}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
