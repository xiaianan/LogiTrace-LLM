#!/usr/bin/env bash
# 压力测试：仅 ddr4_spot_price，长训、较高 LR，不 resume，观察预测是否接近「横线」贴近真值。
# 用法: bash scripts/stress_spot_only_15ep.sh
# 可选环境变量: TIME_LLM_MODEL_PATH, BATCH_SIZE, ACCUMULATION_STEPS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
unset OMP_NUM_THREADS 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

_DEFAULT_SNAPSHOT='/root/autodl-tmp/Time-llm-test/D:\huggingface_models/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c'
export TIME_LLM_MODEL_PATH="${TIME_LLM_MODEL_PATH:-$_DEFAULT_SNAPSHOT}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
ACC="${ACCUMULATION_STEPS:-4}"

exec python train.py \
  --llm_model QWEN2_7B_BNB4 \
  --model_path "${TIME_LLM_MODEL_PATH}" \
  --single_var_spot \
  --data Custom_Cleaned \
  --root_path ./timellm-dataset/storage/processed \
  --data_path time_llm_data_cleaned.csv \
  --batch_size "${BATCH_SIZE}" \
  --accumulation_steps "${ACC}" \
  --learning_rate 0.00005 \
  --weight_decay 0.001 \
  --huber_delta 0.1 \
  --train_epochs 15 \
  --patience 5 \
  --num_workers 0 \
  --checkpoint ./checkpoints/timellm_spot_stress_15ep.pth \
  --resume "" \
  "$@"
