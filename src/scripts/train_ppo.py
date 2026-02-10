import os
from stable_baselines3 import PPO
from src.env.gym_env import DrivingEnv

def main() -> None:
    # 1. Preparación de carpetas
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 2. Entorno de entrenamiento (sin render para máxima velocidad)
    env = DrivingEnv()

    # 3. Configuración del modelo PPO
    # He añadido 'tb_log_name' para organizar mejor tus gráficas
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
    print("Iniciando entrenamiento...")
    model.learn(total_timesteps=300000, tb_log_name="PPO_v1_run")
    
    # Guardar el cerebro de la IA
    model.save("models/ppo_driving_car_v1")
    env.close()

    # 5. Fase de Evaluación (Visual)
    print("\nEntrenamiento completado. Iniciando demostración...")
    eval_env = DrivingEnv(render_mode="human")
    # Cargamos el modelo recién guardado en el entorno visual
    model = PPO.load("models/ppo_driving_car_v1", env=eval_env)

    for episode in range(5):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # deterministic=True para ver el comportamiento "final" aprendido
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            
            # El renderizado ahora será fluido gracias a los cambios en gym_env
            done = terminated or truncated
            total_reward += reward
            
        print(f"Episodio {episode + 1} finalizado. Recompensa total: {total_reward:.2f}")

    eval_env.close()

if __name__ == "__main__":
    main()