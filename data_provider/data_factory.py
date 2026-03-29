from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_provider.data_loader import Dataset_Custom_Cleaned


@dataclass
class _Borders:
    border1: int
    border2: int


def _ett_borders(flag: str) -> _Borders:
    train = 12 * 30 * 24
    val = 4 * 30 * 24
    test = 4 * 30 * 24
    if flag == "train":
        return _Borders(0, train)
    if flag == "val":
        return _Borders(train, train + val)
    if flag == "test":
        return _Borders(train + val, train + val + test)
    raise ValueError(f"Unsupported flag: {flag}")


class ETTWindowDataset(Dataset):
    def __init__(self, args, flag: str):
        super().__init__()
        csv_path = Path(args.root_path) / args.data_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 2:
            raise RuntimeError(f"Dataset is empty: {csv_path}")
        # drop first column (date), parse remaining numeric columns
        parsed = []
        for row in rows[1:]:
            parsed.append([float(v) for v in row[1:]])
        data = np.asarray(parsed, dtype=np.float32)

        b = _ett_borders(flag)
        seq_len = int(args.seq_len)
        label_len = int(args.label_len)
        pred_len = int(args.pred_len)
        window_left = b.border1 - seq_len
        if window_left < 0:
            window_left = 0
        self.data_x = data[window_left : b.border2]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.length = max(0, len(self.data_x) - seq_len - pred_len + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]

        # Keep time features as zeros for compatibility.
        x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        y_mark = np.zeros((self.label_len + self.pred_len, 4), dtype=np.float32)

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
        )


def data_provider(args, flag: str) -> Tuple[Dataset, DataLoader]:
    if str(getattr(args, "data", "")).lower() == "custom_cleaned":
        dataset = Dataset_Custom_Cleaned(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            seq_len=int(args.seq_len),
            label_len=int(args.label_len),
            pred_len=int(args.pred_len),
            train_ratio=float(getattr(args, "train_ratio", 0.70)),
            val_ratio=float(getattr(args, "val_ratio", 0.15)),
            apply_revin=bool(getattr(args, "apply_revin_in_dataset", False)),
            single_var_spot=bool(getattr(args, "single_var_spot", False)),
        )
    else:
        dataset = ETTWindowDataset(args, flag)
    shuffle = flag == "train"
    drop_last = flag == "train"
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=int(getattr(args, "num_workers", 0)),
        drop_last=drop_last,
    )
    return dataset, loader
