# ML Assignment

This repository contains a custom Grid world environment (Gymnasium) and training/evaluation scripts using Stable-Baselines3 (PPO and DQN).

The repo includes all items required for Part 4 (Submission):
- Project code for Parts 1–3 (environment, training, evaluation).
- A comprehensive README (this file) explaining setup, training, and evaluation.
- A separate Design Analysis document: see DESIGN_ANALYSIS.md.

## 1) Environment Overview
GridEnv is a simple 2D M×N grid. An agent must reach a goal while avoiding K obstacles.
- Configurable parameters: M (rows), N (cols), K (number of obstacles), B (number of bonuses to collect)
- Action space: Discrete(4): 0=up, 1=down, 2=left, 3=right
- Observation space: flat vector of length 4 + M×N + (M×N if B>0). Encodes: [agent_row, agent_col, goal_row, goal_col, obstacle_bitmap (M×N), bonus_bitmap (M×N; zeros for collected bonuses)]; dtype=int32.
- Rewards: step penalty -0.005; potential shaping ±0.1 when moving closer/farther from goal; +0.2 for collecting a bonus; reaching goal gives +10 only after all bonuses are collected; stepping on goal early gives -0.1; collisions or out-of-bounds give -1 each; timeout (max steps) gives -3.
- Episode end: termination on obstacle, out-of-bounds, or on goal after all bonuses are collected; truncation when max_steps (default 30) is reached.
- Render modes: "ansi" (returns string), "human" (prints to stdout at render_fps=1 by default)

Expanded Environment (Bonus 2): the agent must visit B bonus positions (checkpoints) before reaching the final goal. Bonuses are shown as "B" in the text render and change to "b" once collected. The goal "G" only terminates the episode successfully when all bonuses are collected. This expansion is fully backward-compatible with B=0 (original task).

Environment implementation: grid_env.py (class GridEnv)

## 2) Setup Instructions
You can run locally with Python, or use Docker for a consistent environment.

### 2.1 Local setup (Windows PowerShell shown; Python 3.11 recommended)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.2 Docker setup — build and run the container (recommended for consistency)
The repository includes a standard Dockerfile (capital D) and a docker-compose.yml with helpful services.

Build the image from the project root:
```powershell
# Build the image
docker build -t ml-assignment .
```
Or with Docker Compose:
```powershell
# Build via compose (uses the builder service in the docker-compose.yml)
docker compose --profile build build builder 
```
Run an interactive container and mount the project (PowerShell):
```powershell
# Start a one-off interactive container
# ${PWD} expands to your current path in PowerShell
# The image tag must match what you built above (ml-assignment)
docker run -it --rm -v ${PWD}:/app --workdir /app ml-assignment bash
```
Using Docker Compose services (see docker-compose.yml):
```powershell
# 1) Train agents using the predefined service/command
# (Runs python train_agents.py with defaults specified in compose)
docker compose run --rm train_agents

# 2) Evaluate PPO agent (expects model under .\models\ppo)
docker compose run --rm evaluate_ppo_agent

# 3) Evaluate DQN agent (expects model under .\models\dqn)
docker compose run --rm evaluate_dqn_agent
```
If you prefer to bring services up (and stop them later):
```powershell
# Start the training service (will run and then exit)
docker compose --profile train up --abort-on-container-exit train_agents

# Evaluate PPO (will run and then exit)
docker compose --profile eval up --abort-on-container-exit evaluate_ppo_agent

# Evaluate DQN
docker compose --profile eval up --abort-on-container-exit evaluate_dqn_agent

# When done with any long-running services, stop and remove
docker compose down
```

Inside the container, you can also run training and evaluation manually, for example:
```bash
# Train (shorter debug example) with 2 bonuses
python train_agents.py --M 5 --N 5 --K 3 --B 2 --ppo_timesteps 100000 --dqn_timesteps 100000 --ppo_envs 4 --out_dir models

# Evaluate a saved PPO model (use matching dimensions)
python eval_agent.py --model ./models/ppo/ppo_grid_final.zip --algo ppo --episodes 200 --M 5 --N 5 --K 5 --B 2
```
Notes:
- Models and logs are written under the mounted project directory (./models) so they persist on your host.
- The base image is Ubuntu; the default container command is `bash`. You can override this in docker run/compose as needed.

## 3) Quick Start (Environment only)
Minimal example to interact with the environment:
```python
from grid_env import GridEnv

env = GridEnv(M=5, N=7, n_obstacle=6, n_bonus=2, render_mode="human")  # set n_bonus=0 to disable bonuses
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### 3a) Run the simplest demo loop
If you just want to see the grid update with random actions, run this minimal example. It prints the grid every second and stops when the episode ends.

PowerShell or cmd (runs the same demo):
```powershell
python grid_env.py
```

Or from Python:
```python
import time
from grid_env import GridEnv

# Set B=0 to disable bonus checkpoints; set render_mode="human" to print the grid
env = GridEnv(M=5, N=7, n_obstacle=6, n_bonus=2, render_mode="human")
obs, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    print("\n")
    time.sleep(1)
    action = env.action_space.sample()  # random action: 0=up,1=down,2=left,3=right
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### 3b) Run a quick random-action demo programmatically
You can also call the included helper to step the environment with random actions:

PowerShell or cmd:
```powershell
python -c "from grid_env import random_env_test; random_env_test(m=6, n=6, k=5, b=2, render_mode='human')"
```

Or from Python:
```python
from grid_env import random_env_test

random_env_test(m=6, n=6, k=5, b=2, render_mode="human")
```

## 4) Training Agents (PPO and DQN)
Training is handled by train_agents.py using Stable-Baselines3.

General help:
```powershell
python train_agents.py -h
```

Default behavior trains PPO and then DQN, saving models under .\models\ppo and .\models\dqn respectively.

Common arguments:
- --M, --N, --K: Grid size and number of obstacles (defaults: 6, 6, 5)
- --B: Number of bonus checkpoints to collect before reaching the goal (default: 0 = disabled)
- --ppo_timesteps: Total PPO timesteps (default: 1000000)
- --dqn_timesteps: Total DQN timesteps (default: 1000000)
- --ppo_envs: Number of parallel envs for PPO (default: 8)
- --out_dir: Output directory for models (default: models)

Example: shorter debug run on a small grid
```powershell
python train_agents.py --M 5 --N 5 --K 3 --B 2 --seed 123 --ppo_timesteps 100000 --dqn_timesteps 100000 --ppo_envs 4 --out_dir models
```

Outputs:
- PPO best models and logs: .\models\ppo
- DQN best models and logs: .\models\dqn
- Final model files: ppo_grid_final.zip, dqn_grid_final.zip

Notes:
- PPO uses vectorized envs (DummyVecEnv or SubprocVecEnv) with flattened float32 observations.
- DQN uses a single DummyVecEnv and a replay buffer (see essential_dqn_kwargs in train_agents.py).

## 5) Evaluating a Trained Agent
Use eval_agent.py to evaluate either a PPO or DQN model.

General help:
```powershell
python eval_agent.py -h
```

Required arguments:
- --model: Path to the .zip model to load (e.g., .\models\ppo\ppo_grid_final.zip or a best_model.zip in the same folder)
- --algo: ppo or dqn

Optional arguments:
- --episodes: Number of evaluation episodes (default: 100)
- --M, --N, --K: Environment parameters to evaluate with (should generally match training)

Examples:
```powershell
# Evaluate the final PPO model (with bonuses)
python eval_agent.py --model .\models\ppo\ppo_grid_final.zip --algo ppo --episodes 200 --M 6 --N 6 --K 5 --B 2

# Evaluate the best DQN model saved by EvalCallback (if present)
python eval_agent.py --model .\models\dqn\best_model.zip --algo dqn --episodes 200 --M 6 --N 6 --K 5 --B 2
```

The script reports:
- Average steps per episode
- Average total reward per episode
- Success rate (%) — proportion of episodes with positive total return (ep_reward > 0)

It also writes a JSON record alongside stdout with the metrics and per-episode details:
- ppo_eval_record.json or dqn_eval_record.json (depending on --algo)

## 6) Reproducing a Full Experiment
1. Setup environment (Section 2).
2. Train agents (Section 4) with your chosen hyperparameters.
3. Evaluate models (Section 5) and record metrics.
4. For algorithm comparison, evaluate both PPO and DQN with the same environment parameters and seeds when possible; average over multiple seeds for robustness.

### 6a) Example Results (Baseline task)
For the baseline task without bonuses (B=0), using the default training setup from docker-compose/train_agents (M=6, N=6, K=5, B=0) and evaluating a well-trained PPO model over 100–200 episodes, we observe metrics in the following ranges:
- Average steps per episode: 5–6
- Average total reward per episode: 9.9–10.2
- Success rate: 87%–95%

These results were obtained with PPO and DQN trained for ~1,000,000 timesteps (default) and deterministic evaluation. Your exact numbers will vary with different hyperparameters.

## 7) Project Files
- grid_env.py — Environment implementation (Gymnasium API).
- train_agents.py — Training for PPO and DQN with SB3; saves models and logs.
- eval_agent.py — Evaluation script to compute steps, rewards, success rate.
- requirements.txt — Python dependencies.
- Dockerfile, docker-compose.yml — Containerized environment setup.
- README.md — This document.
- DESIGN_ANALYSIS.md — Design choices, algorithm comparison, and challenges.

## 8) Troubleshooting
- If Stable-Baselines3 or Gymnasium complain about missing packages, re-run: `pip install -r requirements.txt`.
- On Windows PowerShell, activate the venv with `. .venv\Scripts\Activate.ps1` (note the leading dot and space).
- If SubprocVecEnv hangs on Windows, reduce `--ppo_envs` to 1 (uses DummyVecEnv) or run inside Docker/Linux.

## 9) License
This assignment repository is for educational use.
