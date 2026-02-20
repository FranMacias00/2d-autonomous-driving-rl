"""Evaluación visual de un agente PPO entrenado en DrivingEnv con feedback visual."""

from __future__ import annotations
import os
import time
from stable_baselines3 import PPO
from src.env.gym_env import DrivingEnv

MODEL_PATH = "modelo_entrenado.zip"
N_EPISODES = 3

# Colores para la consola
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def _get_end_reason(terminated: bool, truncated: bool, info: dict) -> tuple[str, str]:
    """Retorna el motivo del fin y un mensaje formateado con emoji."""
    event = info.get("event")

    if event == "finish":
        return "finish", f"{Colors.GREEN}{Colors.BOLD}🏁 ¡META ALCANZADA!{Colors.END}"
    
    if event == "off_track":
        return "off_track", f"{Colors.RED}{Colors.BOLD}💥 COLISIÓN{Colors.END}"
    
    if event == "timeout" or truncated:
        return "timeout", f"{Colors.YELLOW}{Colors.BOLD}⏳ TIEMPO AGOTADO{Colors.END}"
    
    return "unknown", "❓ MOTIVO DESCONOCIDO"

def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"{Colors.RED}Error: No se encontró el modelo en '{MODEL_PATH}'{Colors.END}")
        return

    print(f"\n{Colors.BOLD}🚀 Iniciando evaluación de {N_EPISODES} episodios...{Colors.END}\n")
    
    env = DrivingEnv(render_mode="human")
    model = PPO.load(MODEL_PATH, env=env)

    try:
        for episode in range(1, N_EPISODES + 1):
            obs, _ = env.reset()
            steps = 0
            total_reward = 0.0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                steps += 1
                total_reward += reward

                if terminated or truncated:
                    _, formatted_msg = _get_end_reason(terminated, truncated, info)
                    
                    # Separador visual por episodio
                    print("-" * 60)
                    print(f"EPISODIO {episode}")
                    print(f"  ├─ Resultado: {formatted_msg}")
                    print(f"  ├─ Pasos: {steps}")
                    print(f"  └─ Recompensa Total: {total_reward:.2f}")
                    print("-" * 60 + "\n")
                    
                    time.sleep(1) # Pausa breve para poder leer el resultado en consola
                    break
    finally:
        env.close()
        print(f"{Colors.BOLD}✅ Evaluación finalizada.{Colors.END}")

if __name__ == "__main__":
    main()