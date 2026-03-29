#!/usr/bin/env python3
"""
工业级时序清洗管线：将 time_llm_dataset_aligned.csv 转为 Time-LLM 微调可用的日频数据。

用法:
  python preprocess_data.py
  python preprocess_data.py --input path/to/time_llm_dataset_aligned.csv --output path/out.csv

RevIN 说明见模块常量 `REVIN_TRAINING_NOTE` 或 `get_revin_training_note()`。
"""

# RevIN：训练阶段按实例可逆标准化（本 CSV 不落盘，避免与 batch/窗口强耦合）
REVIN_TRAINING_NOTE = """
RevIN (Reverse Instance Normalization)
- Time-LLM 常在每条样本的时间窗口上按变量通道做: x' = (x - mean) / std，预测后再反变换回物理量纲。
- 本清洗脚本仅输出原始/对数/滚动特征；请在训练代码的 forward 或 Dataset 中对
  ddr4_spot_price、dxi_index_log、gap_pct、nand_* 等价尺度敏感通道执行 RevIN。
- 参考: Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting".
"""


def get_revin_training_note() -> str:
    """供训练脚本 import：返回 RevIN 实施说明（多行字符串）。"""
    return REVIN_TRAINING_NOTE.strip()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 对齐数据中的价差列名（现货相对合约缺口百分比）
GAP_PCT_COL = "ddr4_spot_contract_gap_pct"


def load_and_dedupe_daily(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("CSV 必须包含 date 列")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # 截断至日历日（丢弃日内时间 / 毫秒）
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values("date")
    daily = df.groupby("date", as_index=False).last()
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def detect_contract_roll_flag(
    daily: pd.DataFrame,
    price_col: str = "ddr4_contract_price",
    window_start: str = "2024-05-01",
    window_end: str = "2024-07-31",
    jump_threshold: float = 0.20,
) -> pd.Series:
    """
    合约换月：在 2024-06 前后窗口内，若日环比变动超过阈值则标记为 1。
    仅在指定日期窗口内生效，避免其他时段噪声误报。
    """
    d = daily["date"]
    pct = daily[price_col].pct_change().abs()
    in_window = d.between(pd.Timestamp(window_start), pd.Timestamp(window_end))
    flag = ((pct > jump_threshold) & in_window).fillna(False).astype(np.int8)
    return flag


def reindex_calendar_ffill(daily: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """按自然日建立连续索引；数值前向填充；换月标记不传播（缺失日记 0）。"""
    start = daily["date"].min()
    end = daily["date"].max()
    full_idx = pd.date_range(start=start, end=end, freq="D", name="date")
    indexed = daily.set_index("date").reindex(full_idx)
    roll = indexed[flag_col].reindex(full_idx).fillna(0).astype(np.int8)
    numeric_cols = [c for c in indexed.columns if c != flag_col]
    indexed[numeric_cols] = indexed[numeric_cols].ffill()
    indexed[flag_col] = roll
    out = indexed.reset_index()
    return out


def add_gap_pct_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    if GAP_PCT_COL not in df.columns:
        raise ValueError(f"缺少列 {GAP_PCT_COL}，无法计算 gap_pct 移动平均")
    df = df.copy()
    df["gap_pct"] = df[GAP_PCT_COL].astype(float)
    df["gap_pct_ma7"] = df["gap_pct"].rolling(window=7, min_periods=7).mean()
    df["gap_pct_ma30"] = df["gap_pct"].rolling(window=30, min_periods=30).mean()
    return df


def add_dxi_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dxi = df["dxi_index"].astype(float)
    if (dxi <= 0).any():
        bad = (dxi <= 0).sum()
        raise ValueError(f"dxi_index 存在 {bad} 条非正数，无法安全取对数")
    df["dxi_index_log"] = np.log(dxi)
    return df


def print_distribution_report(before: pd.Series, after: pd.Series) -> None:
    print("\n--- dxi_index 分布对比（变换前 vs log 后）---")
    desc_b = before.describe()
    desc_a = after.describe()
    print("变换前 dxi_index:\n", desc_b.to_string())
    print("\n变换后 dxi_index_log:\n", desc_a.to_string())
    try:
        from scipy import stats as st

        print(f"\n偏度 — 变换前: {st.skew(before, nan_policy='omit'):.6f}  |  变换后: {st.skew(after, nan_policy='omit'):.6f}")
        print(f"峰度 — 变换前: {st.kurtosis(before, nan_policy='omit'):.6f}  |  变换后: {st.kurtosis(after, nan_policy='omit'):.6f}")
    except Exception:
        pass


def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    daily = load_and_dedupe_daily(input_path)
    daily["contract_roll_flag"] = detect_contract_roll_flag(daily)
    filled = reindex_calendar_ffill(daily, "contract_roll_flag")
    filled = add_dxi_log(filled)
    filled = add_gap_pct_moving_averages(filled)

    n_roll = int(filled["contract_roll_flag"].sum())
    print(f"合约换月检测：窗口内 |日环比|>20% 的日期数 = {n_roll}（contract_roll_flag=1）")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(output_path, index=False)
    return filled


def main() -> None:
    root = Path(__file__).resolve().parent
    default_in = root / "timellm-dataset/storage/processed/time_llm_dataset_aligned.csv"
    default_out = root / "timellm-dataset/storage/processed/time_llm_data_cleaned.csv"

    ap = argparse.ArgumentParser(description="Time-LLM 对齐数据清洗与特征工程")
    ap.add_argument("--input", type=Path, default=default_in, help="输入 CSV 路径")
    ap.add_argument("--output", type=Path, default=default_out, help="输出 CSV 路径")
    args = ap.parse_args()

    df = run_pipeline(args.input, args.output)

    start = df["date"].min()
    end = df["date"].max()
    print("\n========== 清洗结果 ==========")
    print(f"总行数: {len(df)}")
    print(f"时间跨度: {start.date()}  →  {end.date()}")
    print(f"已保存: {args.output.resolve()}")

    # 分布对比：使用日终 dedupe 后、日历连续化前的 dxi 与最终 log 列对应同一日历长度
    print_distribution_report(df["dxi_index"], df["dxi_index_log"])


if __name__ == "__main__":
    main()
