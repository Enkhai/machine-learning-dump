import argparse
import json
import os
from typing import Callable, Dict, Any

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from grid_env import GridEnv


def make_env(m: int, n: int, k: int, b: int = 0, render_mode: str | None = None) -> Callable[[], gym.Env]:
    """Factory returning a wrapped GridEnv suitable for SB3 (flattened, float32)."""

    def _thunk() -> gym.Env:
        env = GridEnv(m, n, k, b, render_mode)
        env = TimeLimit(env, max_episode_steps=100)
        env = Monitor(env)
        return env

    return _thunk


class ArgsLoggerCallback(BaseCallback):
    def __init__(self, run_args: Dict[str, Any] | None = None, log_hparams: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.run_args = run_args or {}
        self.log_hparams = log_hparams

    def _on_training_start(self) -> None:
        # 1) Record each arg as a SB3 scalar/text; shows up under TensorBoard scalars
        for k, v in self.run_args.items():
            self.logger.record(f"run_args/{k}", float(v) if isinstance(v, (int, float)) else str(v))
        # Ensure the first dump writes them out immediately (appears at step 0)
        self.logger.dump(step=0)

        # 2) Save a JSON copy next to the TensorBoard event files for reproducibility
        log_dir = self.logger.get_dir()
        if log_dir:
            try:
                with open(os.path.join(log_dir, "run_args.json"), "w", encoding="utf-8") as f:
                    json.dump(self.run_args, f, indent=2)
            except Exception as e:
                if self.verbose:
                    print(f"Could not write run_args.json: {e}")

        # 3) Optionally log as TensorBoard HParams for the HParams plugin (requires torch TB)
        if self.log_hparams and log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                # Write to a separate directory to avoid conflicts with other callbacks/loggers
                hparams_dir = os.path.join(log_dir, "hparams")
                os.makedirs(hparams_dir, exist_ok=True)
                writer = SummaryWriter(hparams_dir)
                # HParams require at least one metric; we add a dummy that won’t be plotted
                metrics = {"hparams/dummy_metric": 0.0}
                # Convert all values to TB-friendly types
                hparams = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in self.run_args.items()}
                writer.add_hparams(hparams, metrics)
                writer.flush()
                writer.close()
            except Exception as e:
                if self.verbose:
                    print(f"HParams logging skipped: {e}")

        return None

    def _on_step(self) -> bool:
        return True


ppo_kwargs = dict(
    learning_rate=3e-4,
    n_steps=256,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=0.2,
    normalize_advantage=True,
    ent_coef=0.03,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=None,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    device="cpu",
)


def train_ppo(total_timesteps: int,
              m: int,
              n: int,
              k: int,
              b: int,
              n_envs: int,
              out_dir: str,
              render_mode: str = "ansi") -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Vectorized envs for PPO
    vec_env = SubprocVecEnv([make_env(m, n, k, b, render_mode) for _ in range(n_envs)]) if n_envs > 1 else DummyVecEnv(
        [make_env(m, n, k, b, render_mode)])

    # Separate eval env
    eval_env = DummyVecEnv([make_env(m, n, k, b, render_mode)])

    args_cb = ArgsLoggerCallback(
        run_args=dict(
            algo="PPO",
            total_timesteps=total_timesteps,
            M=m, N=n, K=k, B=b,
            n_envs=n_envs,
            out_dir=out_dir,
            render_mode=render_mode,
            **{f"ppo.{k}": v for k, v in ppo_kwargs.items() if k in [
                "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "device"]
               }
        ),
        log_hparams=True,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([args_cb, eval_cb])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(out_dir, "tb"),
        **ppo_kwargs
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name="ppo_run",
        reset_num_timesteps=True
    )

    final_path = os.path.join(out_dir, "ppo_grid_final.zip")
    model.save(final_path)
    vec_env.close()
    eval_env.close()
    return final_path


dqn_kwargs = dict(
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=20_000,
    batch_size=128,
    tau=0.005,
    gamma=0.99,
    target_update_interval=2_000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.4,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    max_grad_norm=10,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    device="cpu",
)


def train_dqn(total_timesteps: int,
              m: int,
              n: int,
              k: int,
              b: int,
              out_dir: str,
              render_mode: str = "ansi") -> str:
    os.makedirs(out_dir, exist_ok=True)

    env = DummyVecEnv([make_env(m, n, k, b, render_mode)])
    eval_env = DummyVecEnv([make_env(m, n, k, b, render_mode)])

    args_cb = ArgsLoggerCallback(
        run_args=dict(
            algo="DQN",
            total_timesteps=total_timesteps,
            M=m, N=n, K=k, B=b,
            out_dir=out_dir,
            render_mode=render_mode,
            # Select only flat/numeric-friendly keys from dqn_kwargs
            **{f"dqn.{k}": v for k, v in dqn_kwargs.items() if k in [
                "learning_rate", "buffer_size", "learning_starts", "batch_size",
                "tau", "gamma", "target_update_interval", "train_freq",
                "gradient_steps", "exploration_fraction", "exploration_initial_eps",
                "exploration_final_eps", "max_grad_norm", "device"
            ]}
        ),
        log_hparams=True,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    callback_list = CallbackList([args_cb, eval_cb])

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(out_dir, "tb"),
        **dqn_kwargs
    )
    model.learn(total_timesteps=total_timesteps,
                callback=callback_list,
                tb_log_name="dqn_run",
                reset_num_timesteps=True
                )

    final_path = os.path.join(out_dir, "dqn_grid_final.zip")
    model.save(final_path)
    env.close()
    eval_env.close()
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Train PPO and DQN on GridEnv")
    parser.add_argument("--M", type=int, default=6, help="Grid rows")
    parser.add_argument("--N", type=int, default=6, help="Grid cols")
    parser.add_argument("--K", type=int, default=5, help="Number of obstacles")
    parser.add_argument("--B", type=int, default=0, help="Number of bonus checkpoints to collect before goal")
    parser.add_argument("--ppo_timesteps", type=int, default=1_000_000, help="Total timesteps for PPO")
    parser.add_argument("--dqn_timesteps", type=int, default=1_000_000, help="Total timesteps for DQN")
    parser.add_argument("--ppo_envs", type=int, default=8, help="Number of parallel envs for PPO")
    parser.add_argument("--out_dir", type=str, default=os.path.join("models"), help="Output directory for models")

    args = parser.parse_args()

    ppo_dir = os.path.join(args.out_dir, "ppo")
    dqn_dir = os.path.join(args.out_dir, "dqn")

    print(f"Training PPO on {args.M}x{args.N} grid with K={args.K} obstacles and B={args.B} bonuses...")
    ppo_path = train_ppo(args.ppo_timesteps, args.M, args.N, args.K, args.B, args.ppo_envs, ppo_dir)
    print(f"PPO saved to: {ppo_path}")

    print(f"Training DQN on {args.M}x{args.N} grid with K={args.K} obstacles and B={args.B} bonuses...")
    dqn_path = train_dqn(args.dqn_timesteps, args.M, args.N, args.K, args.B, dqn_dir)
    print(f"DQN saved to: {dqn_path}")


if __name__ == "__main__":
    main()
