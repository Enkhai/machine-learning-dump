from gym.envs.registration import register
from .sudoku import SudokuEnv, VecSudokuEnv
from .connect4 import Connect4Env, VecConnect4Env

register(id='Sudoku-v0', entry_point='gym_sudoku.envs:SudokuEnv')
register(id='VecSudoku-v0', entry_point='gym_sudoku.envs:VecSudokuEnv')
register(id='Connect4-v0', entry_point='gym_sudoku.envs:Connect4Env')
register(id='VecConnect4Env-v0', entry_point='gym_sudoku.envs:VecConnect4Env')
