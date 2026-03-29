#!/usr/bin/env bash
# Time-LLM 微调：清洗后的 DDR4 / DXI 日频数据（AutoDL / 单卡）
# 显存不足请改用 --llm_model QWEN2_7B_BNB4 并设置本地模型路径；A100 可尝试 QWEN2_7B（FP16）
set -euo pipefail

# 避免 OpenMP/libgomp 对异常/空字符串 OMP_NUM_THREADS 报错；可按机器再调大
unset OMP_NUM_THREADS 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# 本地 HF 快照（find 实测完整权重所在路径；可覆盖为 export TIME_LLM_MODEL_PATH=...
_DEFAULT_QWEN_SNAPSHOT='/root/autodl-tmp/Time-llm-test/D:\huggingface_models/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c'
export TIME_LLM_MODEL_PATH="${TIME_LLM_MODEL_PATH:-$_DEFAULT_QWEN_SNAPSHOT}"
# 使用本地快照时跳过 Hub 网络探测（仍须保证快照内文件完整）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# batch_size：~24GB 显存紧张时用 2；显存充足再提高 BATCH_SIZE
BATCH="${BATCH_SIZE:-2}"
# 梯度累积：有效 batch ≈ BATCH × ACCUMULATION_STEPS（默认 2×4=8）
ACC="${ACCUMULATION_STEPS:-4}"

python train.py \
  --task_name short_term_forecast \
  --data Custom_Cleaned \
  --model_path "${TIME_LLM_MODEL_PATH}" \
  --root_path ./timellm-dataset/storage/processed \
  --data_path time_llm_data_cleaned.csv \
  --seq_len 90 \
  --label_len 45 \
  --pred_len 7 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --d_model 32 \
  --d_ff 128 \
  --n_heads 8 \
  --patch_len 16 \
  --stride 8 \
  --llm_model QWEN2_7B_BNB4 \
  --llm_dim 3584 \
  --llm_layers 6 \
  --contract_roll_var_idx 5 \
  --batch_size "${BATCH}" \
  --accumulation_steps "${ACC}" \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 8 \
  --huber_delta 1.0 \
  --cosine_t_max 100 \
  --checkpoint ./checkpoints/timellm_best.pth \
  --target_dims 0,1 \
  "$@"

echo "Finished. Best weights: ./checkpoints/timellm_best.pth"
