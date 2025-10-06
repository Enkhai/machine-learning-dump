import time
from abc import ABC
from typing import Optional

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean


class DeviceAlternatingOnPolicyAlgorithm(OnPolicyAlgorithm, ABC):
    """
    An on-policy algorithm abstract class that subclasses OnPolicyAlgorithm and alternates between collecting rollouts
    on the CPU and training the policy on the GPU by default.\n
    Set `dva=False` during initialization to disable the device-alternating property.\n
    Other arguments and keyworded arguments remain the same.
    """

    def __init__(self, *args, dva=True, **kwargs):
        super(DeviceAlternatingOnPolicyAlgorithm, self).__init__(*args, **kwargs)
        self.dva = dva

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        # Lines marked with # ++ are lines that have been added

        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            if self.dva:  # ++
                self.device = "cpu"  # ++
                self.policy.cpu()  # ++
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            if self.dva:  # ++
                self.device = "cuda"  # ++
                self.policy.cuda()  # ++
            self.train()

        callback.on_training_end()

        return self


class DeviceAlternatingPPO(PPO, DeviceAlternatingOnPolicyAlgorithm):
    """
    A PPO algorithm variant that alternates between collecting rollouts
    on the CPU and training the policy on the GPU by default.\n
    Set `dva=False` during initialization to disable the device-alternating property.\n
    Other arguments and keyworded arguments remain the same.
    """
    pass


class DeviceAlternatingDQN(DQN):
    def __init__(self, *args, dva=True, **kwargs):
        super(DeviceAlternatingDQN, self).__init__(*args, **kwargs)
        self.dva = dva

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "run",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            if self.dva:  # ++
                self.device = "cpu"  # ++
                self.replay_buffer.device = "cpu"  # ++
                self.policy.cpu()  # ++

            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    if self.dva:  # ++
                        self.device = "cuda"  # ++
                        self.replay_buffer.device = "cuda"  # ++
                        self.policy.cuda()  # ++
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self
