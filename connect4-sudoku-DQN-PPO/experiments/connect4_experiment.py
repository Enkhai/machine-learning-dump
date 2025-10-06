from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from algorithms import DeviceAlternatingDQN
from envs import VecConnect4Env
from policies import Connect4CnnMlpDQNPolicy


model_name = "Connect4CnnMlp"

if __name__ == '__main__':
    num_instances = 8
    gamma = 0.95
    total_timesteps = 1_500_000

    env = VecMonitor(VecConnect4Env(num_instances))

    model = DeviceAlternatingDQN(
        Connect4CnnMlpDQNPolicy,
        env,
        batch_size=50_000,
        gamma=gamma,
        exploration_fraction=0.85,
        tensorboard_log="../bin",
        verbose=1,
        device='cpu',
    )

    callbacks = [CheckpointCallback(20480, "../models_bin/" + model_name, model_name)]

    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                tb_log_name=model_name)
    model.save("models_bin/" + model_name + "_" + str(total_timesteps) + "_final")

    env.close()
