# Design Analysis

This document outlines the final design decisions for the grid environment, reward shaping, the algorithm comparison (PPO vs. DQN), and the main challenges we faced along with the mitigations. It reflects the design we used in experiments and the constraints we worked under.

## 1) Environment Analysis

### 1.1 Observation: Flat Vector Representation
- Structure (indices within a 1D vector):
  - [0–1]: agent position (x, y)
  - [2–3]: goal position (x, y)
  - [4 … 4+M*N-1]: obstacle occupancy flags (row-major order)
  - Optional bonuses (if present): [4+M*N … 4+2*(M*N)-1]
- Why a flat vector?
  - Avoids constructing complex and lengthy observation objects.
  - Summarizes the observation efficiently with a fixed, simple layout.
  - Avoids the need for CNN encoders and complex policies.
  - Reduces the rollout space and simplifies the implementation around standard MLP policies.

### 1.2 Dynamic Representation
- The bonus portion of the vector is omitted when no bonuses exist.
- This dynamic layout helps both DQN and PPO converge in settings without bonuses (fewer irrelevant inputs).

### 1.3 “No Bonus 2” Step
- We did not proceed with a second training phase including bonuses due to time constraints.
- The environment remains ready to include bonuses, but the main experiments reported here are without bonuses.

## 2) Reward Shaping

### 2.1 Reward Components
- Goal reward: +10.0
- Potential reward: ±0.1 (based on Manhattan distance change to goal)
- Bonus reward: +0.2 (if bonuses are enabled)
- Step penalty: −0.005 per step
- Collision penalty: −1.0
- Out-of-bounds penalty: −1.0
- Early goal penalty: −0.1 (guards against degenerate early termination patterns)
- Timeout penalty: −3.0 (applied when the episode times out)

### 2.2 Rationale and Adjustments
- We explicitly discouraged policies that merely avoid collisions without reaching the goal:
  - Increased the goal reward from 1 to 10 to emphasize reaching the goal.
  - Added a timeout penalty (default max steps 30) to punish non-progressing trajectories.
  - Added a potential-based shaping term (±0.1 per step) tied to Manhattan distance change.
  - Reduced the step penalty from −0.01 to −0.005 to avoid over-penalizing cautious but progressing behavior.
- Early goal and bonus rewards are not fully optimized because we did not complete the “Bonus 2” training step.

## 3) Algorithm Comparison

### 3.1 PPO
- Produces a working solution relatively easily.
- Requires less hyperparameter tuning.
- Was able to work even without the potential reward shaping in some runs.

### 3.2 DQN
- More difficult to tune hyperparameters; typically needs more training time.
- Benefited significantly from the added potential reward shaping, increased goal reward, and the timeout penalty to converge properly.

## 4) Challenges

### 4.1 Training and Hyperparameter Optimization
- DQN adjustments that helped:
  - Reduced learning rate.
  - Increased `learning_starts`.
  - Soft target network updates aided convergence considerably.
  - Lower `exploration_final_eps` and a longer exploration fraction.
  - A deeper MLP policy (3 layers with 256, 256, 128) improved learning stability.
- PPO tweaks:
  - Largely worked out-of-the-box.
  - Added additional clipping to the value function.
  - Slightly increased the entropy coefficient to encourage exploration.

### 4.2 Experiment Tracking and Logging
- Implemented an `ArgsLoggerCallback` to record hyperparameters for each run.
- Logged to TensorBoard with hparams summaries.
- Saved run arguments to JSON for post-hoc analysis.

### 4.3 Docker and Environment Management
- Requirements installation can be lengthy; the Docker builder service may appear slow.
- Python version mismatches required adjusting dependencies in `requirements.txt`.
  - The bundled DQN model cannot be loaded across SB3 version mismatches.
  - Workaround: run the `train_agents` service before `evaluate_dqn_agent` to regenerate/update the model with the current versions.
- We built training and evaluation services that can run one after the other in Docker (see `docker-compose.yml`).

### 4.4 Seeding and Reproducibility
- We did not wire end-to-end deterministic seeding across the environment, vectorized workers, and the training libraries in this iteration.
- Reason: given time constraints, priority was placed on reward shaping, hyperparameter tuning, and Dockerizing the workflow. In practice, achieving true determinism across Gymnasium envs, SB3 vector environments, NumPy/PyTorch RNGs, and Windows/Docker execution can be brittle; partial seeding often gives a false sense of reproducibility. We deferred this to avoid misleading results and focus on core functionality.
- Implication: runs may vary slightly between executions; reported metrics were averaged over multiple runs to mitigate variance.
- What proper seeding would require (future work): propagate a single `seed` to (a) env reset and obstacle/bonus sampling, (b) SB3’s `set_random_seed` and policy initializers, (c) NumPy/Python/PyTorch RNGs (CPU and CUDA), and (d) all VecEnv workers for PPO.

## 5) Practical Notes
- The observation’s flat vector form pairs well with SB3 MLP policies.
- The dynamic omission of bonuses reduces input dimensionality when unused, improving learning stability.
- Reward shaping (especially potential and timeout) was critical to make DQN reliable on this task.

## 6) Future Work
- Complete the “Bonus 2” phase and re-tune the early goal/bonus rewards.
- Implement end-to-end deterministic seeding across env/training/evaluation (single CLI seed propagated to Gymnasium envs, SB3, NumPy/Python/PyTorch, and all VecEnv workers), and document reproducibility guarantees and caveats.
- Explore CNN encoders only if the grid becomes substantially larger; otherwise MLPs remain sufficient.
- Consider curriculum learning and automated HP search (Optuna) to streamline DQN tuning.
- Improve cross-version portability of saved models (pin SB3 and dependency versions per image).
