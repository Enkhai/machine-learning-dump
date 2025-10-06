from typing import Tuple

import torch as th
import torch.nn.functional as F
from stable_baselines3.dqn.policies import QNetwork
from torch import nn


class Connect4CnnMlpNetwork(nn.Module):
    def __init__(self):
        super(Connect4CnnMlpNetwork, self).__init__()

        self.cnn = nn.Conv2d(1, 4, 4, padding='valid')
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(48, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear_out = nn.Linear(128, 64)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = F.relu(self.cnn(x))
        out = F.relu(self.linear1(self.flat(out)))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        return F.relu(self.linear_out(out))


class Connect4CnnMlpACNetwork(nn.Module):
    def __init__(self):
        super(Connect4CnnMlpACNetwork, self).__init__()
        self.actor, self.critic = Connect4CnnMlpNetwork(), Connect4CnnMlpNetwork()
        self.latent_dim_pi, self.latent_dim_vf = 64, 64

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.actor(x), self.critic(x)

    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        return self.actor(x)

    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        return self.critic(x)


class Connect4CnnMlpQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super(Connect4CnnMlpQNetwork, self).__init__(*args, **kwargs)
        self.q_net = Connect4CnnMlpNetwork()
        self.q_net.linear_out = nn.Linear(self.q_net.linear_out.in_features, self.action_space.n)
        self.features_extractor = nn.Identity()
