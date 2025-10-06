import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(self,
                 M=6,
                 N=6,
                 n_obstacle=5,
                 n_bonus=0,
                 render_mode=None,
                 goal_reward: float = 10.0,
                 potential_reward: float = 0.1,
                 bonus_reward: float = 0.2,
                 step_penalty: float = 0.005,
                 collision_penalty: float = 1.0,
                 out_of_bounds_penalty: float = 1.0,
                 early_goal_penalty: float = 0.1,
                 timeout_penalty: float = 3.0,
                 render_fps=1,
                 max_steps=30
                 ):
        super().__init__()
        self.M, self.N = M, N
        self.n_obstacle = n_obstacle
        self.n_bonus = n_bonus

        self.render_mode = render_mode

        self.goal_reward = goal_reward
        self.potential_reward = potential_reward
        self.bonus_reward = bonus_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.early_goal_penalty = early_goal_penalty
        self.timeout_penalty = timeout_penalty

        self.metadata["render_fps"] = render_fps
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)  # 0 up 1 down 2 left 3 right
        obs_len = 4 + M * N + (M * N if n_bonus else 0)  # agent(2) goal(2) obstacles(MN) bonus(MN)
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_len,), dtype=np.int32)

        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (self.M - 1, self.N - 1)
        self.obstacle_pos: List[Tuple[int, int]] = []
        self.bonus_pos: List[Tuple[int, int]] = []  # remaining bonus cells
        self.bonus_visited: List[bool] = []

        self.step_count = 0

        # pre-compute row-major index helper
        self._rc_to_idx = lambda r, c: r * self.N + c

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # sample unique free cells
        all_cells = [(r, c) for r in range(self.M) for c in range(self.N)]
        chosen = self.np_random.choice(len(all_cells), size=2 + self.n_obstacle + self.n_bonus, replace=False)
        cells = [all_cells[i] for i in chosen]
        self.agent_pos = cells[0]
        self.goal_pos = cells[1]
        self.obstacle_pos = set(cells[2:2 + self.n_obstacle])
        self.bonus_pos = cells[2 + self.n_obstacle:2 + self.n_obstacle + self.n_bonus]
        self.bonus_visited = [False] * self.n_bonus

        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        obs_size = 4 + self.M * self.N + (self.M * self.N if self.n_bonus else 0)

        obs = np.zeros(obs_size, dtype=np.int32)
        obs[0] = self.agent_pos[0]
        obs[1] = self.agent_pos[1]
        obs[2] = self.goal_pos[0]
        obs[3] = self.goal_pos[1]
        # obstacle bitmap
        for r, c in self.obstacle_pos:
            obs[4 + self._rc_to_idx(r, c)] = 1
        # bonus bitmap (zero if collected)
        for i, (r, c) in enumerate(self.bonus_pos):
            if not self.bonus_visited[i]:
                obs[4 + self.M * self.N + self._rc_to_idx(r, c)] = 1
        return obs

    def step(self, action):
        assert self.action_space.contains(action)

        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        terminated = truncated = False
        reward = -self.step_penalty  # step penalty
        info = {}

        self.step_count += 1

        # max steps
        if self.step_count >= self.max_steps:
            truncated = True
            reward = -self.timeout_penalty
            info["max_steps_reached"] = True
        # out of bounds
        elif not (0 <= new_r < self.M and 0 <= new_c < self.N):
            terminated = True
            reward = -self.out_of_bounds_penalty
            info["out_of_bounds"] = True
        # obstacle
        elif (new_r, new_c) in self.obstacle_pos:
            terminated = True
            reward = -self.collision_penalty
            info["collision"] = True
        else:
            # bonus collection
            for i, (br, bc) in enumerate(self.bonus_pos):
                if not self.bonus_visited[i] and (new_r, new_c) == (br, bc):
                    self.bonus_visited[i] = True
                    reward += self.bonus_reward
                    info["bonus_collected"] = i

            # compute manhattan distance to goal
            # if distance closer than previous agent position, reward +0.1
            dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            new_dist = abs(new_r - self.goal_pos[0]) + abs(new_c - self.goal_pos[1])
            if new_dist < dist:
                reward += self.potential_reward
            elif new_dist > dist:
                reward -= self.potential_reward

            # goal reached only after all bonuses
            if (new_r, new_c) == self.goal_pos:
                terminated = True
                if all(self.bonus_visited):
                    reward += self.goal_reward
                    info["goal_reached"] = True
                else:
                    reward -= self.early_goal_penalty
                    info["early_goal"] = True

            self.agent_pos = (new_r, new_c)

        info["bonuses_remaining"] = len(self.bonus_pos) - sum(self.bonus_visited)
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ascii()
        if self.render_mode == "human":
            print(self._render_ascii())
            time.sleep(1 / self.metadata["render_fps"])
        return None

    # ------- console colour board (human) -------
    def _board_symbols(self):
        """Return M×N list-of-lists with plain chars."""
        grid = [["." for _ in range(self.N)] for _ in range(self.M)]
        for r, c in self.obstacle_pos:           grid[r][c] = "X"
        for i, (r, c) in enumerate(self.bonus_pos):
            grid[r][c] = "b" if self.bonus_visited[i] else "B"
        grid[self.goal_pos[0]][self.goal_pos[1]] = "G"
        r, c = self.agent_pos
        grid[r][c] = "A"
        return grid

    def _render_ascii(self):
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DARK = "\033[2m"
        BLUE = "\033[34m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        GREY = "\033[90m"
        ON_WHITE = "\033[47m"

        color_map = {
            "A": f"{BOLD}{BLUE}",
            "G": f"{BOLD}{GREEN}",
            "X": f"{ON_WHITE}{GREY}",
            "B": f"{BOLD}{YELLOW}",
            "b": f"{DARK}{YELLOW}",
        }

        board_str = ""
        for row in self._board_symbols():
            line = ""
            for ch in row:
                if ch in color_map:
                    line += f"{color_map[ch]} {ch} {RESET}"
                else:
                    line += " . "
            board_str += line + "\n"

        return board_str


def random_env_test(m=6, n=6, k=5, b=2, render_mode="human"):
    env = GridEnv(m, n, k, b, render_mode)
    terminated = truncated = False
    while not (terminated or truncated):
        print("\n")
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {['up', 'down', 'left', 'right'][action]}, "
              f"Reward: {reward:.2f}, "
              f"Terminated: {terminated}, "
              f"Truncated: {truncated}, "
              f"Info: {info}")
    env.close()


__all__ = ["GridEnv", "random_env_test"]

if __name__ == "__main__":
    random_env_test()
