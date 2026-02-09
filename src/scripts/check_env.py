"""Quick smoke test for the DrivingEnv Gymnasium wrapper."""

from __future__ import annotations

from src.env.gym_env import DrivingEnv


def main() -> int:
    env = DrivingEnv()
    try:
        for episode in range(1, 4):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            last_info = info
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(
                f"Episode {episode}: reward={total_reward:.2f}, event={last_info.get('event')}"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
