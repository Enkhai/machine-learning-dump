import argparse
import json
from datetime import datetime
from typing import Tuple, Literal

import numpy as np
from stable_baselines3 import PPO, DQN

from grid_env import GridEnv


def evaluate(model_path: str,
             algo: Literal["ppo", "dqn"],
             episodes: int,
             M: int,
             N: int,
             K: int,
             B: int = 0) -> Tuple[float, float, float, list, list, list]:
    env = GridEnv(M, N, K, B)

    if algo.lower() == "ppo":
        model = PPO.load(model_path, device="cpu", verbose=1)
    elif algo.lower() == "dqn":
        model = DQN.load(model_path, device="cpu", verbose=1)
    else:
        raise ValueError("algo must be 'ppo' or 'dqn'")

    total_rewards = []
    total_steps = []
    successes = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
            ep_reward += float(reward)
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        # Success criterion: reached the goal -> positive terminal reward
        successes.append(1.0 if ep_reward > 0 else 0.0)

    env.close()

    avg_steps = float(np.mean(total_steps)) if total_steps else 0.0
    avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0

    return avg_steps, avg_reward, success_rate, total_rewards, total_steps, successes


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on GridEnv (with optional bonuses)")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model .zip")
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn"], required=True, help="Algorithm type")
    parser.add_argument("--episodes", type=int, default=100, help="Number of eval episodes")
    parser.add_argument("--M", type=int, default=6, help="Grid rows")
    parser.add_argument("--N", type=int, default=6, help="Grid cols")
    parser.add_argument("--K", type=int, default=5, help="Number of obstacles")
    parser.add_argument("--B", type=int, default=0, help="Number of bonus checkpoints to collect before goal")

    args = parser.parse_args()

    (avg_steps,
     avg_reward,
     success_rate,
     total_rewards,
     total_steps,
     successes) = evaluate(
        model_path=args.model,
        algo=args.algo,
        episodes=args.episodes,
        M=args.M,
        N=args.N,
        K=args.K,
        B=args.B
    )

    print(f"Average steps per episode: {avg_steps:.2f}")
    print(f"Average total reward per episode: {avg_reward:.3f}")
    print(f"Success rate: {success_rate * 100:.1f}%")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "algo": args.algo,
        "params": {
            "episodes": args.episodes,
            "M": args.M,
            "N": args.N,
            "K": args.K,
            "B": args.B,
        },
        "metrics": {
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
        },
        "per_episode": {
            "rewards": total_rewards,
            "steps": total_steps,
            "success": successes,
        }
    }

    with open(f"{args.algo}_eval_record.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
