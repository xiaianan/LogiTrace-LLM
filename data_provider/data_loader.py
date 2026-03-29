"""
自定义数据集：基于 time_llm_data_cleaned.csv 的多变量时序，用于 Time-LLM 微调。

列约定（顺序固定）：
  目标: ddr4_spot_price, ddr4_spot_contract_gap_pct
  协变量: dxi_index_log, gap_pct_ma7, gap_pct_ma30, contract_roll_flag
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COLS = ("ddr4_spot_price", "ddr4_spot_contract_gap_pct")
COVARIATE_COLS = ("dxi_index_log", "gap_pct_ma7", "gap_pct_ma30", "contract_roll_flag")
ALL_COLS: Tuple[str, ...] = TARGET_COLS + COVARIATE_COLS
# 仅 ddr4_spot_price（消融 / 压力测试：无协变量、无第二目标）
SPOT_ONLY_COLS: Tuple[str, ...] = ("ddr4_spot_price",)

# 与 TimeLLM 中 contract_roll_var_idx 对齐：最后一列为合约换月示性变量
CONTRACT_ROLL_COL_INDEX = len(TARGET_COLS) + COVARIATE_COLS.index("contract_roll_flag")

# 默认在 __getitem__ 中对「价格 + 价差 + 对数指数」做实例级标准化（可选）
DEFAULT_REVIN_COLS = (0, 1, 2)


def revin_instance_norm(
    x: np.ndarray,
    col_indices: Sequence[int],
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RevIN：对指定通道在时间维上减均值除标准差。
    返回 (变换后的 x, mean_per_ch, std_per_ch)，后者 shape (n_channels,) 便于训练后反归一化分析。
    """
    x = np.asarray(x, dtype=np.float64).copy()
    means = np.zeros(len(col_indices), dtype=np.float64)
    stds = np.zeros(len(col_indices), dtype=np.float64)
    for k, c in enumerate(col_indices):
        ch = x[:, c]
        m = float(np.nanmean(ch))
        s = float(np.nanstd(ch)) + eps
        means[k] = m
        stds[k] = s
        x[:, c] = (ch - m) / s
    return x.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


class Dataset_Custom_Cleaned(Dataset):
    """
    读取 time_llm_data_cleaned.csv，按时间切分 train/val/test，滑动窗口生成样本。

    若 ``apply_revin=True``，在取出的窗口上对 ``revin_cols`` 指定通道做实例 RevIN；
    模型侧 ``layers.StandardNorm.Normalize`` 仍会对 **全部** 通道再做一遍可逆标准化，
    因此微调时建议 ``apply_revin=False``（默认），仅保留本函数作为与训练代码对齐的预留接口。
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        data_path: str,
        flag: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        apply_revin: bool = False,
        revin_cols: Sequence[int] = DEFAULT_REVIN_COLS,
        single_var_spot: bool = False,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.flag = flag
        self.apply_revin = bool(apply_revin)
        self.single_var_spot = bool(single_var_spot)
        if self.single_var_spot:
            self.revin_cols = (0,)
        else:
            self.revin_cols = tuple(int(c) for c in revin_cols)

        csv_path = Path(root_path) / data_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"未找到数据文件: {csv_path}")

        df = pd.read_csv(csv_path)
        use_cols = SPOT_ONLY_COLS if self.single_var_spot else ALL_COLS
        missing = [c for c in use_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}，当前列为 {list(df.columns)}")

        df = df.sort_values("date").reset_index(drop=True)
        data = df[list(use_cols)].astype(np.float64)
        data = data.ffill().bfill()
        if data.isnull().any().any():
            raise RuntimeError("ffill/bfill 后仍存在 NaN，请检查源数据")

        self.data_x = data.values.astype(np.float32)
        n = len(self.data_x)
        if n < self.seq_len + self.pred_len:
            raise ValueError(
                f"序列长度不足: n={n}, 需要至少 seq_len+pred_len={self.seq_len + self.pred_len}"
            )

        tr = float(train_ratio)
        vr = float(val_ratio)
        if tr <= 0 or vr <= 0 or tr + vr >= 1.0:
            raise ValueError("train_ratio + val_ratio 必须小于 1，且均为正")

        i_train = int(tr * n)
        i_val = int((tr + vr) * n)
        i_train = max(i_train, self.seq_len + self.pred_len)
        i_val = max(i_val, i_train + self.seq_len + self.pred_len)
        i_val = min(i_val, n)

        if flag == "train":
            self.data = self.data_x[:i_train]
        elif flag == "val":
            self.data = self.data_x[i_train:i_val]
        elif flag == "test":
            self.data = self.data_x[i_val:]
        else:
            raise ValueError(f"flag 必须是 train/val/test，收到: {flag}")

        self.length = max(0, len(self.data) - self.seq_len - self.pred_len + 1)
        if self.length <= 0:
            raise ValueError(
                f"{flag} 集在切分后样本数为 0：len(segment)={len(self.data)}, "
                f"seq_len={self.seq_len}, pred_len={self.pred_len}"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end].copy()
        seq_y = self.data[r_begin:r_end].copy()

        if self.apply_revin:
            seq_x, _, _ = revin_instance_norm(seq_x, self.revin_cols)

        x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        y_mark = np.zeros((self.label_len + self.pred_len, 4), dtype=np.float32)

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )
