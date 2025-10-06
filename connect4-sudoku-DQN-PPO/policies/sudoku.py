import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from models.sudoku import SudokuMixerACNetwork, SudokuCnnACNetwork


class SudokuMixerACPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(SudokuMixerACPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SudokuMixerACNetwork()

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error


class SudokuCnnACPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(SudokuCnnACPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SudokuCnnACNetwork()

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error
