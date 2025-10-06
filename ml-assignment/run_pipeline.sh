#!/usr/bin/env bash
set -euo pipefail

echo "Starting training pipeline..."

# Allow basic overrides via env vars
EPISODES=${EPISODES:-100}
M=${M:-6}
N=${N:-6}
K=${K:-5}
B=${B:-0}
PPO_ENVS=${PPO_ENVS:-8}
PPO_STEPS=${PPO_STEPS:-1000000}
DQN_STEPS=${DQN_STEPS:-1000000}
OUT_DIR=${OUT_DIR:-models}

# Train agents
python3 train_agents.py \
  --M "$M" --N "$N" --K "$K" --B "$B" \
  --ppo_envs "$PPO_ENVS" \
  --ppo_timesteps "$PPO_STEPS" \
  --dqn_timesteps "$DQN_STEPS" \
  --out_dir "$OUT_DIR"

# Evaluate PPO
PPO_MODEL="$OUT_DIR/ppo/ppo_grid_final.zip"
if [ -f "$PPO_MODEL" ]; then
  echo "Evaluating PPO model at $PPO_MODEL ..."
  python3 eval_agent.py --model "$PPO_MODEL" --algo ppo --episodes "$EPISODES" --M "$M" --N "$N" --K "$K" --B "$B"
else
  echo "Warning: PPO model not found at $PPO_MODEL" >&2
fi

# Evaluate DQN
DQN_MODEL="$OUT_DIR/dqn/dqn_grid_final.zip"
if [ -f "$DQN_MODEL" ]; then
  echo "Evaluating DQN model at $DQN_MODEL ..."
  python3 eval_agent.py --model "$DQN_MODEL" --algo dqn --episodes "$EPISODES" --M "$M" --N "$N" --K "$K" --B "$B"
else
  echo "Warning: DQN model not found at $DQN_MODEL" >&2
fi

echo "Pipeline finished. Artifacts in $OUT_DIR and eval records in *_eval_record.json"
