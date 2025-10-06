from typing import Tuple

import torch as th
import torch.nn.functional as F
from torch import nn


class SudokuMixerNetwork(nn.Module):
    def __init__(self):
        super(SudokuMixerNetwork, self).__init__()

        self.linear = nn.Linear(9, 32)
        self.linear_transpose = nn.Linear(9, 32)
        self.cnn = nn.Conv2d(1, 1, 3, padding='valid')
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(900, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear_out = nn.Linear(128, 128)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = F.relu(self.linear(x)).transpose(2, 3)  # column
        out = F.relu((self.linear_transpose(out))).transpose(2, 3)  # row
        out = F.leaky_relu(self.cnn(out))  # grid
        out = F.relu(self.linear1(self.flat(out)))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        return F.relu(self.linear_out(out))  # latent


class SudokuMixerACNetwork(nn.Module):

    def __init__(self):
        super(SudokuMixerACNetwork, self).__init__()

        self.actor = SudokuMixerNetwork()
        self.critic = SudokuMixerNetwork()

        self.latent_dim_pi, self.latent_dim_vf = 128, 128

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.actor(x), self.critic(x)

    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        return self.actor(x)

    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        return self.critic(x)


class SudokuCnnetwork(nn.Module):
    def __init__(self):
        super(SudokuCnnetwork, self).__init__()

        self.cnn1 = nn.Conv2d(1, 3, 3)
        self.cnn2 = nn.Conv2d(3, 8, 3)
        self.cnn3 = nn.Conv2d(8, 20, 3)
        self.flat = nn.Flatten()
        self.lin_out = nn.Linear(180, 64)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = F.relu(self.cnn1(x))
        out = F.relu(self.cnn2(out))
        out = F.relu(self.cnn3(out))
        out = F.relu(self.flat(out))
        return F.relu((self.lin_out(out)))


class SudokuCnnACNetwork(nn.Module):
    def __init__(self):
        super(SudokuCnnACNetwork, self).__init__()

        self.actor, self.critic = SudokuCnnetwork(), SudokuCnnetwork()

        self.latent_dim_pi, self.latent_dim_vf = 64, 64

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.actor(x), self.critic(x)

    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        return self.actor(x)

    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        return self.critic(x)
