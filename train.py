#!/usr/bin/env python3
"""
Time-LLM 微调入口（单卡 / 无 DeepSpeed），面向 time_llm_data_cleaned 小样本场景：
Huber Loss、Cosine Annealing、早停，最优权重保存至 ./checkpoints/timellm_best.pth
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models import TimeLLM
from models.TimeLLM import align_timellm_auxiliary_modules, uses_quantized_llm_backbone


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_target_dims(s: str) -> list:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def add_smoothness_penalty(
    outputs: torch.Tensor,
    base_loss: torch.Tensor,
    alpha_smooth: float,
    smooth_channel: int,
) -> torch.Tensor:
    """total = Huber + alpha * mean(|Δ pred|) 沿时间步，抑制预测形状剧烈起伏。"""
    if alpha_smooth <= 0 or outputs.size(1) <= 1:
        return base_loss
    ch = int(smooth_channel)
    if ch < 0 or ch >= outputs.size(-1):
        raise IndexError(
            f"smooth_channel={ch} 超出 outputs 通道数 {outputs.size(-1)}"
        )
    diff = outputs[:, 1:, ch] - outputs[:, :-1, ch]
    smooth_loss = torch.mean(torch.abs(diff))
    return base_loss + float(alpha_smooth) * smooth_loss


def evaluate(
    args,
    device: torch.device,
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    target_dims: list,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float()

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len :, :]
            target = batch_y[:, -args.pred_len :, :]
            if target_dims is not None:
                outputs = outputs[:, :, target_dims]
                target = target[:, :, target_dims]
            loss = criterion(outputs, target)
            loss = add_smoothness_penalty(
                outputs,
                loss,
                float(getattr(args, "alpha_smooth", 0.0) or 0.0),
                int(getattr(args, "smooth_channel", 0)),
            )
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


class BestCheckpoint:
    def __init__(self, save_path: Path, patience: int, min_delta: float = 0.0):
        self.save_path = save_path
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"[BestCheckpoint] val_loss={val_loss:.6f} -> saved {self.save_path}")
        else:
            self.counter += 1
            print(f"[EarlyStopping] {self.counter}/{self.patience} (best={self.best:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True


def get_parser() -> argparse.ArgumentParser:
    """
    构建训练用 `ArgumentParser`（仅定义参数，不解析）。

    其他脚本（如 `eval_and_report.py`）可通过本函数取回同一套参数定义，
    再 `parser.add_argument(...)` 扩展评估专用项，或 `parse_known_args` 做子集解析。
    """
    p = argparse.ArgumentParser(description="Time-LLM fine-tune (Custom cleaned CSV)")
    p.add_argument("--task_name", type=str, default="short_term_forecast")
    p.add_argument("--model_id", type=str, default="timellm_custom")
    p.add_argument("--data", type=str, default="Custom_Cleaned")
    p.add_argument("--root_path", type=str, default="./timellm-dataset/storage/processed")
    p.add_argument("--data_path", type=str, default="time_llm_data_cleaned.csv")
    p.add_argument("--features", type=str, default="M", help="M: 多变量预测多变量（损失可只取 target_dims）")
    p.add_argument("--target_dims", type=str, default="0,1", help="Huber 监督的通道下标（对应两个目标变量）")
    p.add_argument("--seq_len", type=int, default=90)
    p.add_argument("--label_len", type=int, default=45)
    p.add_argument("--pred_len", type=int, default=7)
    p.add_argument("--enc_in", type=int, default=6)
    p.add_argument("--dec_in", type=int, default=6)
    p.add_argument("--c_out", type=int, default=6)
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--reprogramming_dropout",
        type=float,
        default=None,
        help="Reprogramming 注意力 dropout；默认与 --dropout 相同（0.1）",
    )
    p.add_argument("--embed", type=str, default="timeF")
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--output_attention", action="store_true", default=False)
    p.add_argument("--patch_len", type=int, default=16)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--prompt_domain", type=int, default=0)
    p.add_argument("--llm_model", type=str, default="QWEN2_7B_BNB4", help="QWEN2_7B / QWEN2_7B_BNB4 等")
    p.add_argument("--llm_dim", type=int, default=3584)
    p.add_argument("--llm_layers", type=int, default=6)
    p.add_argument("--model_path", type=str, default="", help="本地 Qwen 目录或 HF id")
    p.add_argument("--contract_roll_var_idx", type=int, default=5, help="contract_roll_flag 在通道中的下标")
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--train-patch-embedding", dest="train_patch_embedding", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train-reprogramming", dest="train_reprogramming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force_single_gpu_map", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数；有效 batch ≈ batch_size × accumulation_steps（显存不足时配合小 batch）",
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--train_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5, help="小样本建议 1e-5")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument(
        "--alpha_smooth",
        type=float,
        default=0.0,
        help="预测序列平滑惩罚：total = Huber + alpha * mean(|Δpred|)；0 关闭",
    )
    p.add_argument(
        "--smooth_channel",
        type=int,
        default=0,
        help="平滑项作用在 target_dims 切片后的通道下标（默认 0=ddr4_spot_price）",
    )
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--apply_revin_in_dataset", action="store_true", default=False)
    p.add_argument(
        "--single_var_spot",
        action="store_true",
        default=False,
        help="仅使用 ddr4_spot_price 单通道（无协变量）；自动 enc_in=1 且关闭 contract_roll",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="可选：从已有 .pth 加载 LoRA/reprogramming 等适配权重；留空则从头训练适配器",
    )
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--checkpoint", type=str, default="./checkpoints/timellm_best.pth")
    p.add_argument("--cosine_t_max", type=int, default=None, help="默认等于 train_epochs")
    p.add_argument("--cosine_eta_min", type=float, default=1e-8)
    return p


def build_args(argv: Optional[list[str]] = None):
    """
    解析命令行参数并返回 `Namespace`。

    - `argv is None`：使用 `sys.argv`（与直接 `python train.py ...` 行为一致）。
    - `argv` 为列表时：用于单元测试或程序化调用，例如 ``build_args(['--train_epochs', '1'])``。
    """
    return get_parser().parse_args(argv)


def collect_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


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
    for k, v in filtered.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            compatible[k] = v
    model.load_state_dict(compatible, strict=False)
    print(f"[Checkpoint] Loaded {len(compatible)} keys from {ckpt_path}", flush=True)
    return len(compatible)


def main():
    args = build_args()
    if getattr(args, "reprogramming_dropout", None) is None:
        args.reprogramming_dropout = args.dropout
    if getattr(args, "single_var_spot", False):
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.target_dims = "0"
        args.contract_roll_var_idx = None
    set_seed(args.seed)
    target_dims = parse_target_dims(args.target_dims)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if getattr(args, "model_path", None) in (None, "") and os.environ.get("TIME_LLM_MODEL_PATH"):
        args.model_path = os.environ["TIME_LLM_MODEL_PATH"]

    train_loader = data_provider(args, "train")[1]
    val_loader = data_provider(args, "val")[1]

    model = TimeLLM.Model(args)
    if uses_quantized_llm_backbone(getattr(args, "llm_model", "")):
        align_timellm_auxiliary_modules(model)
    else:
        model = model.float().to(device)

    resume_path = str(getattr(args, "resume", "") or "").strip()
    if resume_path:
        load_filtered_checkpoint(model, Path(resume_path), device)
    else:
        print("[Train] No --resume: training adapters from Qwen2 backbone init only.", flush=True)

    params = collect_trainable_params(model)
    if not params:
        raise RuntimeError("没有可训练参数：请检查 LoRA / 冻结设置")
    n_trainable = sum(p.numel() for p in params)
    print(f"[Model] Trainable parameters (total elements): {n_trainable:,}", flush=True)
    optim = Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    t_max = args.cosine_t_max or args.train_epochs
    scheduler = CosineAnnealingLR(optim, T_max=t_max, eta_min=args.cosine_eta_min)

    criterion = nn.HuberLoss(delta=args.huber_delta)
    assert isinstance(criterion, nn.HuberLoss), "损失函数须为 HuberLoss"
    assert float(criterion.delta) == float(args.huber_delta), "Huber delta 与配置不一致"
    accum = max(1, int(getattr(args, "accumulation_steps", 1)))
    eff_bs = args.batch_size * accum
    print(
        f"[Loss] nn.HuberLoss(delta={criterion.delta})  alpha_smooth={getattr(args, 'alpha_smooth', 0.0)} "
        f"(smooth_ch={getattr(args, 'smooth_channel', 0)})  weight_decay={args.weight_decay} "
        f"reprogramming_dropout={getattr(args, 'reprogramming_dropout', args.dropout)}"
    )
    print(
        f"[Train] batch_size={args.batch_size} accumulation_steps={accum} "
        f"effective_batch≈{eff_bs}",
        flush=True,
    )

    checkpoint = BestCheckpoint(Path(args.checkpoint), patience=args.patience)

    for epoch in range(args.train_epochs):
        model.train()
        epoch_losses = []
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.train_epochs}")
        for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float()

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -args.pred_len :, :]
            target = batch_y[:, -args.pred_len :, :]
            if target_dims:
                outputs = outputs[:, :, target_dims]
                target = target[:, :, target_dims]

            loss = criterion(outputs, target)
            loss = add_smoothness_penalty(
                outputs,
                loss,
                float(getattr(args, "alpha_smooth", 0.0) or 0.0),
                int(getattr(args, "smooth_channel", 0)),
            )
            (loss / accum).backward()
            epoch_losses.append(loss.item())

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                optim.step()
                optim.zero_grad(set_to_none=True)

            pbar.set_postfix(loss=float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = evaluate(args, device, model, val_loader, criterion, target_dims)
        print(
            f"Epoch {epoch+1}/{args.train_epochs} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | lr={optim.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        checkpoint(val_loss, model)
        if checkpoint.early_stop:
            print("Early stopping.")
            break

    print(f"Done. Best val_loss={checkpoint.best:.6f}, checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
