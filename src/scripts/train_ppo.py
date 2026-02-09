import os

from stable_baselines3 import PPO

from src.env.gym_env import DrivingEnv


def main() -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DrivingEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./logs/",
    )

    model.learn(total_timesteps=300000)
    model.save("models/ppo_driving_car_v1")
    env.close()

    eval_env = DrivingEnv(render_mode="human")
    model = PPO.load("models/ppo_driving_car_v1", env=eval_env)

    for episode in range(5):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {episode + 1} total reward: {total_reward}")

    eval_env.close()


if __name__ == "__main__":
    main()
