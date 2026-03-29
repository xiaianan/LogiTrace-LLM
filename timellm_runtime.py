"""
Time-LLM online inference for workflow_orchestrator / Streamlit.

- Data: default `time_llm_data_cleaned.csv` (training-aligned); aligned CSV lacks dxi_index_log / gap_pct_ma* etc.
- Weights: `TIME_LLM_CHECKPOINT` or `--checkpoint` default `./checkpoints/timellm_best.pth`
- Backbone: **`TIME_LLM_MODEL_PATH` is required** (local Qwen2-7B-Instruct dir or HF id). No auto-discovery and no default hub id when unset.
- Mirror: if you use an HF id and huggingface.co is slow, set `TIME_LLM_HF_MIRROR=1` (hf-mirror.com) or `HF_ENDPOINT=...`.
- Optional: `TIME_LLM_FP16_QWEN=1` forces `QWEN2_7B` (FP16, no bitsandbytes 4-bit) if you have enough VRAM or bnb issues persist after `pip install -r requirements.txt`.
- Disable: `TIME_LLM_ORCH_DISABLE=1` → orchestrator uses stub
- Cache: process singleton; `clear_timellm_cache()` or `TIME_LLM_ORCH_RELOAD=1` to reload
"""

from __future__ import annotations

import os


def _apply_hf_hub_mirror_early() -> None:
    """When huggingface.co is unreachable, enable mirror before any hub/transformers download."""
    if os.environ.get("TIME_LLM_HF_MIRROR", "").lower() not in ("1", "true", "yes"):
        return
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


_apply_hf_hub_mirror_early()

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent

_bundle: Optional[Tuple[Any, Any, torch.device]] = None


def clear_timellm_cache() -> None:
    global _bundle
    _bundle = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_model_args(args: Any) -> None:
    from eval_and_report import _ensure_model_args as _e

    _e(args)


def _resolved_under_repo(path_or_empty: Optional[str], train_default: str) -> str:
    """Anchor relative paths to this repo so inference works regardless of process cwd."""
    env = (path_or_empty or "").strip()
    chosen = env if env else train_default
    p = Path(chosen)
    if p.is_absolute():
        return str(p)
    return str((ROOT / p).resolve())


def _prepare_args() -> Any:
    from train import get_parser

    p = get_parser()
    args = p.parse_args([])
    if getattr(args, "reprogramming_dropout", None) is None:
        args.reprogramming_dropout = args.dropout
    if getattr(args, "single_var_spot", False):
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.target_dims = "0"
        args.contract_roll_var_idx = None

    args.root_path = _resolved_under_repo(os.environ.get("TIME_LLM_ROOT"), args.root_path)
    args.data_path = os.environ.get("TIME_LLM_DATA_PATH", args.data_path)
    args.checkpoint = _resolved_under_repo(os.environ.get("TIME_LLM_CHECKPOINT"), args.checkpoint)
    _mp_env = os.environ.get("TIME_LLM_MODEL_PATH")
    if _mp_env and str(_mp_env).strip():
        mp = str(_mp_env).strip()
        mp_path = Path(mp)
        args.model_path = str(mp_path if mp_path.is_absolute() else (ROOT / mp_path).resolve())

    if os.environ.get("TIME_LLM_FP16_QWEN", "").lower() in ("1", "true", "yes"):
        args.llm_model = "QWEN2_7B"

    if os.environ.get("TIME_LLM_SINGLE_VAR_SPOT", "").lower() in ("1", "true", "yes"):
        args.single_var_spot = True
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.target_dims = "0"
        args.contract_roll_var_idx = None

    args.num_workers = 0
    _ensure_model_args(args)
    return args


def _load_bundle() -> Tuple[Any, Any, torch.device]:
    global _bundle
    if os.environ.get("TIME_LLM_ORCH_RELOAD", "").lower() in ("1", "true", "yes"):
        clear_timellm_cache()
    if _bundle is not None:
        return _bundle

    from eval_and_report import load_filtered_checkpoint
    from models import TimeLLM
    from models.TimeLLM import align_timellm_auxiliary_modules, uses_quantized_llm_backbone

    args = _prepare_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Time-LLM checkpoint not found: {ckpt.resolve()}")
    mp = str(getattr(args, "model_path", "") or "").strip()
    if not mp:
        raise ValueError(
            "TIME_LLM_MODEL_PATH is required for Time-LLM inference (no default hub id or auto-discovery). "
            "Export it to your Qwen2-7B-Instruct snapshot directory or HF model id, e.g. same as in scripts/fine_tune.sh."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = TimeLLM.Model(args)
        if uses_quantized_llm_backbone(getattr(args, "llm_model", "")):
            align_timellm_auxiliary_modules(model)
        else:
            model = model.float().to(device)
    except KeyError as exc:
        if str(exc).strip("'\"") == "qwen2" or "qwen2" in str(exc).lower():
            raise RuntimeError(
                "Transformers is too old to load Qwen2 (got KeyError 'qwen2'). "
                "Install transformers>=4.40 (see requirements.txt), e.g. "
                "`pip install 'transformers>=4.40' 'accelerate>=0.30'`."
            ) from exc
        raise
    load_filtered_checkpoint(model, ckpt, device)
    model.eval()
    _bundle = (args, model, device)
    return _bundle


def _load_series_for_timellm(
    args: Any,
) -> Tuple[pd.DataFrame, np.ndarray, Tuple[str, ...]]:
    from data_provider.data_loader import ALL_COLS, SPOT_ONLY_COLS

    use_cols = SPOT_ONLY_COLS if getattr(args, "single_var_spot", False) else ALL_COLS
    csv_path = Path(args.root_path) / args.data_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Time-LLM data CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path).sort_values("date").reset_index(drop=True)
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns for Time-LLM: {missing}")

    data = df[list(use_cols)].astype(np.float64).ffill().bfill()
    if data.isnull().any().any():
        raise RuntimeError("NaN after ffill/bfill in Time-LLM CSV")
    arr = data.values.astype(np.float32)
    n = len(arr)
    seq_len, pred_len = int(args.seq_len), int(args.pred_len)
    if n < seq_len + pred_len:
        raise ValueError(f"Time series too short: n={n}, need seq_len+pred_len={seq_len + pred_len}")
    return df, arr, tuple(use_cols)


def _global_end_idx_bounds(n: int, args: Any) -> Tuple[int, int]:
    seq_len, label_len, pred_len = int(args.seq_len), int(args.label_len), int(args.pred_len)
    imin = max(seq_len - 1, label_len - 1)
    imax = n - pred_len - 1
    return imin, imax


def _batch_at_global_end_idx(
    df: pd.DataFrame,
    arr: np.ndarray,
    use_cols: Tuple[str, ...],
    args: Any,
    device: torch.device,
    global_end_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], Tuple[str, ...]]:
    n = len(arr)
    seq_len, label_len, pred_len = int(args.seq_len), int(args.label_len), int(args.pred_len)
    imin, imax = _global_end_idx_bounds(n, args)
    ge = int(global_end_idx)
    if ge < imin or ge > imax:
        raise ValueError(
            f"global_end_idx={ge} out of valid range [{imin}, {imax}] (n={n})"
        )

    s_begin = ge - seq_len + 1
    s_end = ge + 1
    r_begin = s_end - label_len
    r_end = r_begin + label_len + pred_len
    seq_x = arr[s_begin:s_end]
    seq_y = arr[r_begin:r_end]
    x_mark = np.zeros((seq_len, 4), dtype=np.float32)
    y_mark = np.zeros((label_len + pred_len, 4), dtype=np.float32)

    date_series = df["date"]
    horizon_dates = [str(date_series.iloc[s_end + k]) for k in range(pred_len)]

    bx = torch.from_numpy(seq_x).unsqueeze(0).float().to(device)
    by = torch.from_numpy(seq_y).unsqueeze(0).float().to(device)
    xm = torch.from_numpy(x_mark).unsqueeze(0).float().to(device)
    ym = torch.from_numpy(y_mark).unsqueeze(0).float().to(device)
    return bx, by, xm, ym, horizon_dates, tuple(use_cols)


def _last_window_batch(
    args: Any,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    List[str],
    Tuple[str, ...],
]:
    df, arr, use_cols = _load_series_for_timellm(args)
    n = len(arr)
    ge = n - int(args.pred_len) - 1
    bx, by, xm, ym, horizon_dates, colnames = _batch_at_global_end_idx(
        df, arr, use_cols, args, device, ge
    )
    return bx, by, xm, ym, arr, horizon_dates, colnames


def run_timellm_numpy() -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    One forward pass on the last valid window of the full series.

    Returns
    -------
    pred : (pred_len, c_out) denormalized multi-channel forecast
    last_row : (c_out,) anchor row at global_end_idx = n - pred_len - 1 (not the CSV physical last row)
    args : Namespace
    """
    args, model, device = _load_bundle()
    bx, by, xm, ym, arr, _, _ = _last_window_batch(args, device)
    pred_len = int(args.pred_len)
    label_len = int(args.label_len)

    dec_inp = torch.zeros_like(by[:, -pred_len:, :]).float().to(device)
    dec_inp = torch.cat([by[:, :label_len, :], dec_inp], dim=1).float()

    with torch.no_grad():
        out = model(bx, xm, dec_inp, ym)
    pred = out.detach().float().cpu().numpy()[0]
    ge = len(arr) - int(args.pred_len) - 1
    last_row = arr[ge].copy()
    return pred, last_row, args


@dataclass
class LastWindowPredTruth:
    """Last-window batch aligned with Dataset_Custom_Cleaned: pred steps vs seq_y tail truth."""

    pred: np.ndarray
    true_y: np.ndarray
    horizon_dates: List[str]
    column_names: Tuple[str, ...]
    args: Any


def get_last_window_pred_vs_truth() -> LastWindowPredTruth:
    """Last window: model output vs CSV future pred_len steps (training targets)."""
    args, model, device = _load_bundle()
    bx, by, xm, ym, arr, horizon_dates, column_names = _last_window_batch(args, device)
    pred_len = int(args.pred_len)
    label_len = int(args.label_len)

    dec_inp = torch.zeros_like(by[:, -pred_len:, :]).float().to(device)
    dec_inp = torch.cat([by[:, :label_len, :], dec_inp], dim=1).float()

    with torch.no_grad():
        out = model(bx, xm, dec_inp, ym)
    pred = out.detach().float().cpu().numpy()[0]
    true_y = by[0, -pred_len:, :].detach().float().cpu().numpy()
    return LastWindowPredTruth(
        pred=pred.astype(np.float64),
        true_y=true_y.astype(np.float64),
        horizon_dates=horizon_dates,
        column_names=column_names,
        args=args,
    )


def get_timeline_slots_last_month(args: Optional[Any] = None) -> List[Tuple[int, str]]:
    """
    Streamlit timeline: valid anchor dates in the last ~30 calendar days, else last 31 valid anchors.
    Returns (global_end_idx, YYYY-MM-DD label).
    """
    a = args or _prepare_args()
    df, arr, _ = _load_series_for_timellm(a)
    n = len(arr)
    imin, imax = _global_end_idx_bounds(n, a)
    if imax < imin:
        return []
    candidates = list(range(imin, imax + 1))
    dts = pd.to_datetime(df["date"], errors="coerce")
    last_d = dts.max()
    start_m = last_d - pd.Timedelta(days=30)
    month_idx = [
        i
        for i in candidates
        if pd.notna(dts.iloc[i]) and (start_m <= dts.iloc[i] <= last_d)
    ]
    if not month_idx:
        month_idx = candidates[-31:] if len(candidates) > 31 else candidates
    out: List[Tuple[int, str]] = []
    for i in month_idx:
        di = dts.iloc[i]
        label = di.strftime("%Y-%m-%d") if pd.notna(di) else str(i)
        out.append((i, label))
    return out


@dataclass
class ForecastAtDate:
    """Single Time-LLM forward at history cutoff global_end_idx."""

    global_end_idx: int
    anchor_date: str
    anchor_feature_row: np.ndarray
    pred: np.ndarray
    true_y: np.ndarray
    horizon_dates: List[str]
    column_names: Tuple[str, ...]
    args: Any


def forward_at_global_end_idx(global_end_idx: int) -> ForecastAtDate:
    args, model, device = _load_bundle()
    df, arr, use_cols = _load_series_for_timellm(args)
    bx, by, xm, ym, horizon_dates, column_names = _batch_at_global_end_idx(
        df, arr, use_cols, args, device, global_end_idx
    )
    pred_len = int(args.pred_len)
    label_len = int(args.label_len)
    dec_inp = torch.zeros_like(by[:, -pred_len:, :]).float().to(device)
    dec_inp = torch.cat([by[:, :label_len, :], dec_inp], dim=1).float()
    with torch.no_grad():
        out = model(bx, xm, dec_inp, ym)
    pred = out.detach().float().cpu().numpy()[0]
    true_y = by[0, -pred_len:, :].detach().float().cpu().numpy()
    ge = int(global_end_idx)
    anchor_date = str(df["date"].iloc[ge])
    anchor_row = arr[ge].copy()
    return ForecastAtDate(
        global_end_idx=ge,
        anchor_date=anchor_date,
        anchor_feature_row=anchor_row.astype(np.float64),
        pred=pred.astype(np.float64),
        true_y=true_y.astype(np.float64),
        horizon_dates=horizon_dates,
        column_names=column_names,
        args=args,
    )


def numeric_forecast_from_anchor(
    pred: np.ndarray,
    anchor_row: np.ndarray,
    args: Any,
) -> TimellmNumericForecast:
    last_spot = float(anchor_row[0])
    pred_spot = pred[:, 0].astype(np.float64)
    pct = (pred_spot / max(last_spot, 1e-8) - 1.0) * 100.0
    last_gap: Optional[float] = None
    pred_gap: Optional[np.ndarray] = None
    last_dxi_log: Optional[float] = None
    if anchor_row.shape[0] > 1:
        last_gap = float(anchor_row[1])
        pred_gap = pred[:, 1].astype(np.float64)
    if anchor_row.shape[0] > 2:
        last_dxi_log = float(anchor_row[2])
    return TimellmNumericForecast(
        pred_len=int(args.pred_len),
        last_spot=last_spot,
        pred_spot=pred_spot,
        last_gap_pct=last_gap,
        pred_gap_pct=pred_gap,
        last_dxi_log=last_dxi_log,
        spot_change_pct_min=float(np.min(pct)),
        spot_change_pct_max=float(np.max(pct)),
        spot_change_pct_mean=float(np.mean(pct)),
    )


def orchestrator_forecast_at_end_idx(global_end_idx: int, weeks: int = 8) -> Tuple[Any, ForecastAtDate]:
    fd = forward_at_global_end_idx(global_end_idx)
    num = numeric_forecast_from_anchor(fd.pred, fd.anchor_feature_row, fd.args)
    return numeric_to_orchestrator_forecast(num, weeks_param=weeks), fd


@dataclass
class TimellmNumericForecast:
    """Structured numeric forecast for UI / orchestrator."""

    pred_len: int
    last_spot: float
    pred_spot: np.ndarray
    last_gap_pct: Optional[float]
    pred_gap_pct: Optional[np.ndarray]
    last_dxi_log: Optional[float]
    spot_change_pct_min: float
    spot_change_pct_max: float
    spot_change_pct_mean: float


def compute_numeric_forecast() -> TimellmNumericForecast:
    pred, anchor_row, args = run_timellm_numpy()
    return numeric_forecast_from_anchor(pred, anchor_row, args)


def numeric_to_orchestrator_forecast(
    num: TimellmNumericForecast,
    weeks_param: int = 8,
) -> Any:
    """Map to workflow_orchestrator.TimeLLMForecast (lazy import avoids cycles)."""
    from workflow_orchestrator import TimeLLMForecast

    weeks_param = max(4, min(8, int(weeks_param)))
    horizon_days = num.pred_len
    horizon_weeks = max(1, int(np.ceil(horizon_days / 7)))

    m = num.spot_change_pct_mean
    if m > 0.4:
        dxi_trend = "bullish_rebound"
    elif m < -0.4:
        dxi_trend = "bearish_soft"
    else:
        dxi_trend = "sideways_range"

    expected_change_range_pct = (
        f"{num.spot_change_pct_min:.2f}% ~ {num.spot_change_pct_max:.2f}% "
        f"(DDR4 spot, {horizon_days} steps)"
    )

    parts = [
        f"Time-LLM forecasts the next {horizon_days} steps from the cleaned-series window: ",
        f"last spot {num.last_spot:.4f}, relative move ~{expected_change_range_pct}. ",
    ]
    if num.last_dxi_log is not None:
        parts.append(
            f"Input DXI(log) tail {num.last_dxi_log:.4f} (covariate; backbone does not regress DXI path). "
        )
    if num.last_gap_pct is not None and num.pred_gap_pct is not None:
        parts.append(
            f"Spot-contract gap% tail {num.last_gap_pct:.4f}, forecast band "
            f"[{float(np.min(num.pred_gap_pct)):.4f}, {float(np.max(num.pred_gap_pct)):.4f}]. "
        )
    rationale = "".join(parts)

    return TimeLLMForecast(
        horizon_weeks=min(horizon_weeks, weeks_param),
        dxi_trend=dxi_trend,
        expected_change_range_pct=expected_change_range_pct,
        rationale=rationale,
    )


def try_real_timellm_forecast(weeks: int = 8) -> Any:
    """Returns TimeLLMForecast on success; raises on failure (orchestrator may catch)."""
    num = compute_numeric_forecast()
    return numeric_to_orchestrator_forecast(num, weeks_param=weeks)
