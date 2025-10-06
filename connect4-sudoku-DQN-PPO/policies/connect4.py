import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from models import Connect4CnnMlpACNetwork, Connect4CnnMlpQNetwork


class Connect4CnnMlpACPolicy(ActorCriticPolicy):

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Connect4CnnMlpACNetwork()

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error


class Connect4CnnMlpDQNPolicy(DQNPolicy):

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return Connect4CnnMlpQNetwork(**net_args).to(self.device)
