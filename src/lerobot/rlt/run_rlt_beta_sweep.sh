#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BETAS_STR=${RLT_BETA_VALUES:-"1.0 0.75"}
read -ra BETAS <<< "${BETAS_STR}"

NOISES_STR=${RLT_NOISE_VALUES:-"0.12 0.16"}
read -ra ACTION_NOISES <<< "${NOISES_STR}"

BASE_CMD=(
  python rlt/rlt_training.py
  --policy.path=HuggingFaceVLA/smolvla_libero
  --env.type=libero
  --env.task=libero_spatial
  --eval.batch_size=4
  --eval.n_episodes=10
  --env.task_ids=[4]
)

for beta in "${BETAS[@]}"; do
  for noise in "${ACTION_NOISES[@]}"; do
    echo "========================================================="
    echo "Running RLT training with beta=${beta}, noise=${noise}"
    echo "========================================================="
    RLT_BETA="${beta}" RLT_ACTION_NOISE="${noise}" "${BASE_CMD[@]}" "$@"
    echo
    echo "Finished beta=${beta}, noise=${noise}"
    echo
    sleep 2
  done
done
