#!/usr/bin/env python3
"""
加载 checkpoint（含 Qwen2 微调权重），在测试集上推理；按日历聚合得到 true_values / predicted_values。

水平对齐（默认开启）：无论模型输出何种形状，用首点误差把整条预测拽回真实「起跑线」：
  bias = true_values[0] - predicted_values[0]
  predicted_values = predicted_values + bias
使 predicted_values[0] == true_values[0]（例如现货首点约在 6.x 附近）。绘图与对齐后指标均基于平移后的 predicted_values。

另：在按日聚合之前，对每条推理输出的 pred_len 步序列做线性混合，抑制跨 7 步的夸张弧线：
  pred[i] = pred[i] * (1 - alpha*i) + last_value * (alpha*i)，last_value=pred[0]，默认 alpha=0.05。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from data_provider.data_loader import TARGET_COLS
from models import TimeLLM
from models.TimeLLM import align_timellm_auxiliary_modules, uses_quantized_llm_backbone
from train import get_parser

_CHECKPOINT_KEY_SUBSTRS = ("reprogramming", "roll_embed", "roll_to_patch", "lora")


def filter_trainable_checkpoint(state: dict) -> dict:
    return {
        k: v
        for k, v in state.items()
        if any(s in k for s in _CHECKPOINT_KEY_SUBSTRS)
    }


def load_filtered_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device) -> int:
    state = torch.load(ckpt_path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 应为 state_dict，收到: {type(state)}")
    filtered = filter_trainable_checkpoint(state)
    model_sd = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in filtered.items():
        if k not in model_sd:
            skipped.append(k)
            continue
        if v.shape != model_sd[k].shape:
            skipped.append(f"{k} (ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)})")
            continue
        compatible[k] = v
    model.load_state_dict(compatible, strict=False)
    print(f"[Checkpoint] Loaded {len(compatible)} matching keys.", flush=True)
    if skipped:
        print(f"[Checkpoint] Skipped {len(skipped)} keys (missing or shape mismatch).", flush=True)
    return len(compatible)


def _split_borders(
    n: int, seq_len: int, pred_len: int, train_ratio: float, val_ratio: float
) -> tuple[int, int]:
    tr = float(train_ratio)
    vr = float(val_ratio)
    i_train = int(tr * n)
    i_val = int((tr + vr) * n)
    i_train = max(i_train, seq_len + pred_len)
    i_val = max(i_val, i_train + seq_len + pred_len)
    i_val = min(i_val, n)
    return i_train, i_val


def _apply_linear_horizon_smooth(pred: np.ndarray, alpha: float) -> None:
    """
    对形状 (B, H) 的预测做原地平滑。对每个 batch 行：
    pred[i] = pred[i] * (1 - alpha*i) + last_value * (alpha*i)，last_value = 该行原始 pred[0]。
    H 通常为 pred_len（如 7），使多步预测逐步向首步预测靠拢、曲线更平。
    """
    if alpha <= 0 or pred.size == 0:
        return
    bsz, h = pred.shape[0], pred.shape[1]
    for b in range(bsz):
        row = pred[b].astype(np.float64, copy=True)
        last_value = float(row[0])
        for i in range(h):
            t = min(alpha * float(i), 1.0)
            pred[b, i] = row[i] * (1.0 - t) + last_value * t


def _ensure_model_args(args: Any) -> None:
    """补全 TimeLLM 可能用到的可选字段（旧版 train 解析器未定义时）。"""
    defaults = {
        "prompt_max_length": 384,
        "compact_time_prompt": True,
        "gradient_checkpointing": True,
        "reprogram_alpha": 1.0,
        "patch_embed_scale": 0.1,
        "force_single_gpu_map": False,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def main() -> None:
    parser = get_parser()
    parser.add_argument(
        "--report_png",
        type=str,
        default="evaluation_report.png",
        help="输出预测曲线图路径",
    )
    parser.add_argument(
        "--spot_channel",
        type=int,
        default=0,
        help="ddr4_spot_price 在通道中的下标（默认 0）",
    )
    parser.add_argument(
        "--ma_window",
        type=int,
        default=4,
        help="绘图前对预测序列做滑动平均的窗口（1 表示不平滑）",
    )
    parser.add_argument(
        "--external-align",
        dest="external_align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="首点水平对齐：bias=true_values[0]-predicted_values[0]，predicted_values+=bias（默认开；--no-external-align 关闭）",
    )
    parser.add_argument(
        "--horizon_smooth_alpha",
        type=float,
        default=0.05,
        help="对每条 pred_len 输出：pred[i]=pred[i]*(1-a*i)+pred[0]*(a*i)；0 关闭（默认 0.05）",
    )
    args = parser.parse_args()
    _ensure_model_args(args)
    args.num_workers = 0
    if getattr(args, "reprogramming_dropout", None) is None:
        args.reprogramming_dropout = args.dropout
    if not getattr(args, "model_path", None) and os.environ.get("TIME_LLM_MODEL_PATH"):
        args.model_path = os.environ["TIME_LLM_MODEL_PATH"]
    if getattr(args, "single_var_spot", False):
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.target_dims = "0"
        args.contract_roll_var_idx = None

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"未找到权重: {ckpt.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt.resolve()}")

    csv_path = Path(args.root_path) / args.data_path
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    i_train, i_val = _split_borders(
        n, args.seq_len, args.pred_len, args.train_ratio, args.val_ratio
    )
    dates = df["date"].values
    roll_flags = (
        df["contract_roll_flag"].values.astype(float) if "contract_roll_flag" in df.columns else None
    )

    model = TimeLLM.Model(args)
    if uses_quantized_llm_backbone(getattr(args, "llm_model", "")):
        align_timellm_auxiliary_modules(model)
    else:
        model = model.float().to(device)
    load_filtered_checkpoint(model, ckpt, device)
    model.eval()

    _bs = args.batch_size
    args.batch_size = 1
    _, test_loader = data_provider(args, "test")
    args.batch_size = _bs
    ch = int(args.spot_channel)

    pred_by_date: dict = defaultdict(lambda: [0.0, 0])
    true_by_date: dict = defaultdict(lambda: [0.0, 0])

    test_len = len(test_loader.dataset)
    sample_i = 0
    with torch.no_grad():
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float()

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred = outputs[:, -args.pred_len :, ch].detach().cpu().numpy()
            true = batch_y[:, -args.pred_len :, ch].detach().cpu().numpy()

            a_smooth = float(getattr(args, "horizon_smooth_alpha", 0.05) or 0.0)
            if a_smooth > 0:
                _apply_linear_horizon_smooth(pred, a_smooth)

            bsz = pred.shape[0]
            for b in range(bsz):
                if sample_i >= test_len:
                    break
                s_begin = sample_i
                sample_i += 1
                for k in range(args.pred_len):
                    row = i_val + s_begin + args.seq_len + k
                    if row >= n:
                        continue
                    d = dates[row]
                    pv = float(pred[b, k])
                    tv = float(true[b, k])
                    pred_by_date[d][0] += pv
                    pred_by_date[d][1] += 1
                    true_by_date[d][0] += tv
                    true_by_date[d][1] += 1

    common = sorted(set(pred_by_date.keys()) & set(true_by_date.keys()))
    xs = [pd.Timestamp(d) for d in common]
    true_values = np.array(
        [true_by_date[d][0] / max(1, true_by_date[d][1]) for d in common], dtype=np.float64
    )
    predicted_values = np.array(
        [pred_by_date[d][0] / max(1, pred_by_date[d][1]) for d in common], dtype=np.float64
    )

    n_points = int(len(common))
    mse_raw = float(np.mean((predicted_values - true_values) ** 2)) if n_points else float("nan")
    mae_raw = float(np.mean(np.abs(predicted_values - true_values))) if n_points else float("nan")

    bias: float | None = None
    if args.external_align and n_points > 0:
        bias = float(true_values[0] - predicted_values[0])
        predicted_values = predicted_values + bias

    mse_aligned = (
        float(np.mean((predicted_values - true_values) ** 2))
        if (args.external_align and n_points)
        else float("nan")
    )
    mae_aligned = (
        float(np.mean(np.abs(predicted_values - true_values)))
        if (args.external_align and n_points)
        else float("nan")
    )

    print(f"\n=== Test · {TARGET_COLS[ch]} (channel {ch}) · aggregated by date ===")
    _ha = float(getattr(args, "horizon_smooth_alpha", 0.05) or 0.0)
    if _ha > 0:
        print(f"Horizon linear smooth: alpha={_ha} (pred[i]=pred[i]*(1-a*i)+pred[0]*(a*i), len={args.pred_len})")
    print(f"Points: {n_points}")
    print(f"MSE (raw, before level shift):  {mse_raw:.8f}")
    print(f"MAE (raw, before level shift):  {mae_raw:.8f}")
    if args.external_align and bias is not None:
        print(f"Level align: bias = true_values[0] - predicted_values[0] = {bias:+.6f}")
        print(f"  -> first aligned pred equals true_values[0] = {true_values[0]:.6f}")
        print(f"MSE (after predicted_values += bias): {mse_aligned:.8f}")
        print(f"MAE (after predicted_values += bias): {mae_aligned:.8f}")

    ma_w = max(1, int(getattr(args, "ma_window", 4)))
    y_curve = predicted_values.astype(float).copy()
    if ma_w > 1 and len(y_curve) >= ma_w:
        ser = pd.Series(y_curve, dtype=float)
        y_curve = ser.rolling(window=ma_w, center=True, min_periods=1).mean().to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    ax.plot(xs, true_values, label="True (test, mean over windows)", color="#1f77b4", linewidth=1.8)
    pred_label = "Predicted"
    if ma_w > 1:
        pred_label += f" (MA w={ma_w})"
    if _ha > 0:
        pred_label += f", lin-smooth α={_ha}"
    if args.external_align and bias is not None:
        pred_label += ", level align (start = true[0])"
    ax.plot(xs, y_curve, label=pred_label, color="#ff7f0e", linewidth=1.5, alpha=0.9)
    if roll_flags is not None:
        for ri in range(i_val, n):
            if roll_flags[ri] >= 0.5:
                ax.axvline(x=dates[ri], color="crimson", linestyle="--", alpha=0.35, linewidth=1)
        ax.plot([], [], color="crimson", linestyle="--", label="contract_roll_flag=1")
    title = "DDR4 spot price — test forecast vs truth"
    if args.external_align and bias is not None:
        title += " (level aligned to first point)"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COLS[ch])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    out = Path(args.report_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out.resolve()}")


if __name__ == "__main__":
    main()
