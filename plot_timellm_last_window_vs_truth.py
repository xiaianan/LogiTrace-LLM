#!/usr/bin/env python3
"""
本版 Time-LLM（与 timellm_runtime / orchestrator 同源）在 Dataset_Custom_Cleaned
**全表最后一窗**上的逐步预测 vs 同窗真实 `seq_y` 末 pred_len 步。

与训练一致：`time_llm_data_cleaned.csv`、checkpoint、Qwen 路径由环境变量或 train 默认指定。

用法:
  export TIME_LLM_MODEL_PATH=...
  python plot_timellm_last_window_vs_truth.py --output timellm_last_window_vs_truth.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv()
except ImportError:
    pass

from timellm_runtime import clear_timellm_cache, get_last_window_pred_vs_truth


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default="timellm_last_window_vs_truth.png",
        help="输出 PNG 路径",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="忽略进程内 Time-LLM 缓存，重新加载权重",
    )
    args_ns = p.parse_args()
    if args_ns.reload:
        os.environ["TIME_LLM_ORCH_RELOAD"] = "1"
        clear_timellm_cache()

    r = get_last_window_pred_vs_truth()
    pred, true_y = r.pred, r.true_y
    h, c = pred.shape
    assert true_y.shape == (h, c)

    ckpt = Path(getattr(r.args, "checkpoint", ""))
    data_csv = Path(r.args.root_path) / r.args.data_path

    fig_h = max(4.0, 2.8 * min(c, 6))
    fig, axes = plt.subplots(min(c, 6), 1, figsize=(11, fig_h), dpi=120, sharex=True)
    if c == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for ch in range(min(c, 6)):
        ax = axes_flat[ch]
        mse = float(np.mean((pred[:, ch] - true_y[:, ch]) ** 2))
        mae = float(np.mean(np.abs(pred[:, ch] - true_y[:, ch])))
        x = np.arange(1, h + 1)
        ax.plot(x, true_y[:, ch], "o-", label="True (CSV / Dataset_Custom_Cleaned)", color="#1f77b4", lw=2)
        ax.plot(x, pred[:, ch], "s--", label="Predicted (Time-LLM)", color="#ff7f0e", lw=1.8, alpha=0.9)
        ax.set_ylabel(r.column_names[ch])
        ax.set_title(f"{r.column_names[ch]}  |  MSE={mse:.6f}  MAE={mae:.6f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for j in range(min(c, 6), len(axes_flat)):
        axes_flat[j].set_visible(False)

    axes_flat[-1].set_xlabel("Horizon step (1 … pred_len)")
    axes_flat[-1].set_xticks(np.arange(1, h + 1))
    date_compact = [d[:10] if len(d) >= 10 else d for d in r.horizon_dates]
    axes_flat[-1].set_xticklabels([f"{i}\n{d}" for i, d in enumerate(date_compact, start=1)], fontsize=7)

    title = (
        "Time-LLM vs truth — last window (same indexing as Dataset_Custom_Cleaned)\n"
        f"data={data_csv.name}  checkpoint={ckpt.name}  pred_len={h}"
    )
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out = Path(args_ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
