import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.TimeLLM import Model as TimeLLM
from quick_verify import load_state_dict_compatible, make_quick_timellm_args


def _to_2d_history(arr_like):
    arr = np.asarray(arr_like, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("history must be 1D or 2D numeric array")
    return arr


def _pad_or_trim_history(x, seq_len, n_features):
    t, c = x.shape
    if c < n_features:
        x = np.concatenate([x, np.zeros((t, n_features - c), dtype=np.float32)], axis=1)
    elif c > n_features:
        x = x[:, :n_features]

    if t < seq_len:
        pad_rows = np.repeat(x[:1], repeats=seq_len - t, axis=0)
        x = np.concatenate([pad_rows, x], axis=0)
    elif t > seq_len:
        x = x[-seq_len:]
    return x.astype(np.float32)


def _build_args(repo_root, text_prompt, pred_len, ckpt_meta):
    # No-prompt inference uses a tiny reprogramming gain to avoid semantic interference.
    no_prompt_alpha = 0.01
    default_alpha = float(ckpt_meta.get("reprogram_alpha", 0.3))
    effective_alpha = default_alpha if text_prompt else no_prompt_alpha
    return make_quick_timellm_args(
        repo_root=repo_root,
        dataset_root=repo_root / "dataset" / "ETT-small",
        seq_len=int(ckpt_meta.get("seq_len", 256)),
        label_len=int(ckpt_meta.get("label_len", 48)),
        pred_len=pred_len,
        batch_size=1,
        patch_len=int(ckpt_meta.get("patch_len", 16)),
        stride=int(ckpt_meta.get("stride", 8)),
        prompt_domain=1 if text_prompt else 0,
        content=text_prompt or "",
        llm_model=str(ckpt_meta.get("llm_model", "QWEN2_7B_BNB4")),
        model_path=str(ckpt_meta.get("model_path", "Qwen/Qwen2-7B-Instruct")),
        use_lora=bool(ckpt_meta.get("use_lora", True)),
        train_patch_embedding=bool(ckpt_meta.get("train_patch_embedding", False)),
        train_reprogramming=bool(ckpt_meta.get("train_reprogramming", True)),
        lora_r=int(ckpt_meta.get("lora_r", 8)),
        lora_alpha=int(ckpt_meta.get("lora_alpha", 16)),
        lora_dropout=float(ckpt_meta.get("lora_dropout", 0.05)),
        prompt_max_length=int(ckpt_meta.get("prompt_max_length", 96)),
        content_max_length=int(ckpt_meta.get("content_max_length", 128)),
        force_single_gpu_map=True,
        reprogram_alpha=effective_alpha,
    )


def run_demo(input_json, output_json, ckpt_path=None):
    repo_root = Path(__file__).resolve().parent
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))

    history = _to_2d_history(payload["history"])
    text_prompt = str(payload.get("prompt", ""))
    pred_len = int(payload.get("pred_len", 96))

    ckpt_meta = {}
    ckpt_state = None
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_meta = ckpt
        ckpt_state = ckpt.get("model_state_dict", ckpt)

    args = _build_args(repo_root, text_prompt, pred_len, ckpt_meta)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_np = _pad_or_trim_history(history, seq_len=args.seq_len, n_features=args.enc_in)
    x_enc = torch.from_numpy(x_np).unsqueeze(0).to(device)
    x_mark = torch.zeros((1, args.seq_len, 4), dtype=torch.float32, device=device)

    label_part = x_enc[:, -args.label_len :, :]
    dec_zeros = torch.zeros((1, args.pred_len, args.enc_in), dtype=torch.float32, device=device)
    dec_inp = torch.cat([label_part, dec_zeros], dim=1)
    y_mark = torch.zeros((1, args.label_len + args.pred_len, 4), dtype=torch.float32, device=device)

    model = TimeLLM(args)
    if args.llm_model in {"LLAMA3_8B_GPTQ", "QWEN2_7B_GPTQ_INT4", "QWEN2_7B_BNB4"}:
        model.patch_embedding = model.patch_embedding.to(device)
        model.reprogramming_layer = model.reprogramming_layer.to(device)
        model.output_projection = model.output_projection.to(device)
        model.output_norm = model.output_norm.to(device)
        model.output_smoother = model.output_smoother.to(device)
        model.normalize_layers = model.normalize_layers.to(device)
        if model.mapping_layer is not None:
            model.mapping_layer = model.mapping_layer.to(device)
    else:
        model = model.to(device)

    if ckpt_state is not None:
        load_state_dict_compatible(model, ckpt_state)

    model.eval()
    with torch.no_grad():
        y_hat = model(x_enc.float(), x_mark, dec_inp.float(), y_mark)
        y_hat = y_hat[:, -args.pred_len :, :]
        y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1e4, neginf=-1e4)
        pred = y_hat[0, :, 0].detach().float().cpu().numpy()

    used_fallback = False
    if not np.all(np.isfinite(pred)):
        used_fallback = True
        last_val = float(x_np[-1, 0])
        pred = np.full((args.pred_len,), last_val, dtype=np.float32)

    out = {
        "predictions": pred.tolist(),
        "meta": {
            "pred_len": args.pred_len,
            "seq_len": args.seq_len,
            "features_used": int(args.enc_in),
            "prompt_used": bool(text_prompt),
            "checkpoint": ckpt_path or "None (pure Qwen2 backbone)",
            "fallback_used": used_fallback,
        },
    }
    Path(output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Time-LLM demo forecast API")
    parser.add_argument("--input-json", required=True, help="Path to input json")
    parser.add_argument("--output-json", required=True, help="Path to output json")
    parser.add_argument("--ckpt", default="", help="Optional checkpoint path")
    args = parser.parse_args()
    run_demo(args.input_json, args.output_json, ckpt_path=args.ckpt or None)


if __name__ == "__main__":
    main()

