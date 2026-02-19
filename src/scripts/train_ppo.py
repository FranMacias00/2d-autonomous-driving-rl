import os
import time
import datetime
from stable_baselines3 import PPO
from src.env.gym_env import DrivingEnv

def main() -> None:
    # --- CONFIGURACIÓN DE VERSIÓN Y NOMBRES ---
    # Cambiar el número de versión para cada nueva ejecución para mantener un historial organizado
    VERSION = "v1_9sensors" 
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ppo_driving_car_{VERSION}_{run_id}"
    log_name = f"PPO_{VERSION}_{run_id}"
    # ------------------------------------------

    # 1. Preparación de carpetas
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 2. Entorno de entrenamiento (sin render para máxima velocidad)
    env = DrivingEnv()

    # 3. Configuración del modelo PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./logs/",
    )

    # 4. Fase de Aprendizaje
    print(f"Iniciando entrenamiento: {log_name}...")
    model.learn(total_timesteps=300000, tb_log_name=log_name)
    
    # Guardar el modelo usando el nombre dinámico
    model_path = f"models/{model_name}"
    model.save(model_path)
    env.close()

    # 5. Fase de Evaluación (Visual)
    print(f"\nEntrenamiento completado. Cargando: {model_path}")
    eval_env = DrivingEnv(render_mode="human")
    
    # Cargamos el modelo recién guardado de forma dinámica
    model = PPO.load(model_path, env=eval_env)

    for episode in range(5):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # Control de velocidad para humanos (60 FPS)
            time.sleep(1/60)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episodio {episode + 1} finalizado. Recompensa total: {total_reward:.2f}")

    eval_env.close()

if __name__ == "__main__":
    main()