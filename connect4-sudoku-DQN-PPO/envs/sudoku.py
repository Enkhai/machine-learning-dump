import sys
from functools import partial
from itertools import product
from typing import Union

import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box
from stable_baselines3.common.vec_env import DummyVecEnv


class SudokuEnv(Env):
    _win = 0
    _unfinished = 1
    _loss = 2

    _empty_state = np.full((9, 9), np.NaN)

    def __init__(self, n_empty=40):
        self.n_empty = n_empty
        self.action_space = MultiDiscrete([9, 9, 9])  # row, column, number
        # image obs: channel, row, column
        self.observation_space = Box(low=0, high=1, shape=(1, 9, 9), dtype=np.uint8)
        self.state = None
        self._make_initial_state()
        self.reset()

    def _make_image_obs(self):
        obs = self.state.copy()[None]
        obs = np.where(np.isnan(obs), 0, obs)
        return obs / 9

    def _make_initial_state(self):
        self.initial_state = self.get_solutions(self._empty_state.copy())[0]
        xs = np.random.randint(0, 9, self.n_empty)
        ys = np.random.randint(0, 9, self.n_empty)
        self.initial_state[xs, ys] = np.NaN

    def get_solutions(self,
                      state: Union[None, np.ndarray] = None,
                      n_solutions: int = 1):
        if state is None:
            state = self.state

        status = self._check_status(state)
        if status == self._win:
            return [state]
        if status == self._loss:
            return None

        # 1st nan index
        nan_rows, nan_cols = np.where(np.isnan(state))
        nan_idx = [nan_rows[0], nan_cols[0]]

        solutions = []
        values = np.arange(1, 10)
        np.random.shuffle(values)

        for value in values:
            state_branch = state.copy()
            state_branch[nan_idx[0], nan_idx[1]] = value
            solution_branches = self.get_solutions(state_branch, n_solutions - len(solutions))
            if solution_branches is not None:
                solutions += solution_branches
            if len(solutions) == n_solutions:
                return solutions
        else:
            return solutions

    def _check_status(self, state=None) -> int:
        if state is None:
            state = self.state
        nan_mask = np.isnan(state)

        # non-validity = loss
        # row check
        for i in range(9):
            r = state[i][~nan_mask[i]]
            if np.unique(r).size < r.size:
                return self._loss
        # column check
        for i in range(9):
            c = state[:, i][~nan_mask[:, i]]
            if np.unique(c).size < c.size:
                return self._loss
        # sub-grid check
        for i, j in product(*[[0, 3, 6]] * 2):
            subgrid = state[i:i + 3, j:j + 3][~nan_mask[i:i + 3, j:j + 3]]
            if np.unique(subgrid).size < subgrid.size:
                return self._loss

        # validity + all-filled = win
        if not nan_mask.sum():
            return self._win

        # validity = unfinished
        return self._unfinished

    def step(self, action: np.ndarray):
        """
        :param action: Multi-discrete action within ranges [0, 0, 0] and [8, 8, 8]
        :return: observation numpy array, reward scalar, episode done boolean, info dict
        """

        # case: position already filled
        if not np.isnan(self.state[action[0], action[1]]):
            return self._make_image_obs(), -1, False, {'episode': None, 'is_success': False}

        # case: valid action
        self.state[action[0], action[1]] = action[2] + 1  # plus one for range 1-9

        status = self._check_status()
        state = self._make_image_obs()

        # valid action sub-cases: win, unfinished, loss
        if status == self._win:
            return state, 10, True, {'episode': None, 'is_success': True}
        elif status == self._unfinished:
            return state, 1, False, {'episode': None, 'is_success': False}
        elif status == self._loss:
            return state, -10, True, {'episode': None, 'is_success': False}

    def render(self, **kwargs):
        sys.stdout.write('\n-------------------------\n')
        for i in range(9):
            sys.stdout.write('| ')
            r = self.state[i]
            for j in range(9):
                if np.isnan(r[j]):
                    sys.stdout.write(' ')
                else:
                    sys.stdout.write(str(int(r[j])))
                sys.stdout.write(' ')
                if (j + 1) % 3 == 0:
                    sys.stdout.write('| ')
            sys.stdout.write('\n')
            if (i + 1) % 3 == 0:
                sys.stdout.write('-------------------------\n')

    def reset(self):
        self.state = self.initial_state.copy()
        return self.state.copy()[None]


class VecSudokuEnv(DummyVecEnv):
    def __init__(self, n_envs=8):
        super(VecSudokuEnv, self).__init__([partial(SudokuEnv) for _ in range(n_envs)])
