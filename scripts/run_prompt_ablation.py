#!/usr/bin/env python3
"""
Run UC1–UC4 with V1 / V2 / V3 user prompt template (custom-mode-equivalent inputs).
Writes ../v1结果.md, ../v2结果.md, or ../v3结果.md. Requires GLM_API_KEY in .env.

Usage:
  python scripts/run_prompt_ablation.py --prompt v1
  python scripts/run_prompt_ablation.py --prompt v2
  python scripts/run_prompt_ablation.py --prompt v3
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()

import workflow_orchestrator as wo
from zhipuai import ZhipuAI

PROMPTS = {
    "v1": """You are a semiconductor market expert. Using your general knowledge, write a detailed analysis of the DRAM market and procurement strategy.

Inputs for reference (you may go beyond them):
- Outlook: {horizon_weeks} weeks, about {expected_change_range_pct}
- Levels: DDR4 spot {ddr4_spot_price}, DXI {dxi_index}, gap {gap_pct}%
- News: supply '{production_news}', demand '{demand_news}'
- Tags: supply_trend={supply_trend}, demand_trend={demand_trend}
- Notes: {logic_lines}

Be persuasive and confident. Predict specific future spot prices if helpful. Long answer welcome.""",
    "v2": """Role: Supply-chain analyst.

Use the data below. Give procurement advice in a Markdown table with columns: Risk (1-5), Procurement ratio, Rationale.

Data:
- Horizon: {horizon_weeks} weeks; stated range: {expected_change_range_pct}
- DDR4 spot {ddr4_spot_price}; DXI {dxi_index}; Gap% {gap_pct}
- News — supply: {production_news}; demand: {demand_news}
- Tags: supply_trend={supply_trend}; demand_trend={demand_trend}

Consider these logic bullets:
{logic_lines}

Keep the answer under 300 words. Be professional.""",
    "v3": """Role: Industrial memory procurement advisor (conservative, audit-friendly).

Hard rules:
1) Use ONLY the numbers and claims explicitly present in the fields below for factual statements. Do not invent prices, dates, vendor actions, or statistics not stated in the news text.
2) The string "{expected_change_range_pct}" is a USER-DEFINED scenario band for discussion, NOT a verified model forecast unless the text itself states otherwise. Refer to it as a scenario assumption.
3) If supply or demand news is vague, unverified, or missing detail, say so briefly; do not fill gaps with fabricated specifics.
4) You MUST follow the pre-reasoning logic block line-by-line. If any line begins with "[CONFLICT]", prioritize defensive procurement: smaller tranches, MOQ discipline, working-capital caution—do not recommend aggressive buy-ups without strong hedging language.

Inputs:
- Scenario horizon (weeks): {horizon_weeks}
- User scenario band (spot % vs anchor): {expected_change_range_pct}
- Live levels: DDR4 spot {ddr4_spot_price}; DXI {dxi_index}; spot vs contract gap {gap_pct}%
- News — supply: {production_news}
- News — demand: {demand_news}
- Parsed tags: supply_trend={supply_trend}; demand_trend={demand_trend}
- Pipeline / forecast note (Time-LLM or custom; may be brief): {forecast_rationale}

Pre-reasoning logic (do not contradict):
{logic_lines}

Output (Markdown):
- Table: [Risk 1-5] | [Suggested procurement ratio or stance] | [3 short bullets: rationale tied to Gap%, DXI band, and tags]
- One line: "Grounding check" — list which input fields you relied on (spot, DXI, gap, scenario band, which sentence of news).

Max ~250 words. Chinese or English.""",
}

OUTPUT_FILES = {"v1": "v1结果.md", "v2": "v2结果.md", "v3": "v3结果.md"}

USE_CASES = [
    {
        "id": "UC1",
        "name": "深贴水 + 明确减产",
        "check": "C4",
        "date": "2026-03-01",
        "ddr4": 6.18,
        "dxi": 51032.0,
        "gap": -28.5,
        "horizon_weeks": 8,
        "expected_range": "-4.0% ~ +6.0% (DDR4 spot vs anchor, custom scenario, ~8w)",
        "production_news": "Samsung and SK Hynix announce ~15% DRAM output cut for Q2.",
        "demand_news": "Handset pull-in remains soft; server demand flat QoQ.",
        "sentiment": {"supply_trend": "CUT", "demand_trend": "WEAK"},
    },
    {
        "id": "UC2",
        "name": "信息极稀疏",
        "check": "C4",
        "date": "2026-03-01",
        "ddr4": 6.50,
        "dxi": 55000.0,
        "gap": -8.0,
        "horizon_weeks": 12,
        "expected_range": "-1.0% ~ +1.0% (DDR4 spot vs anchor, custom scenario, ~12w)",
        "production_news": "Supply news unavailable.",
        "demand_news": "Demand unclear.",
        "sentiment": {"supply_trend": "STABLE", "demand_trend": "NEUTRAL"},
    },
    {
        "id": "UC3",
        "name": "冲突路由压力测",
        "check": "C3",
        "date": "2026-03-01",
        "ddr4": 5.95,
        "dxi": 48000.0,
        "gap": -12.0,
        "horizon_weeks": 8,
        "expected_range": "bullish upside scenario: +6% ~ +12% vs spot (custom assumption, ~8w)",
        "production_news": "Multiple fabs rumored to extend downtime.",
        "demand_news": "End demand weak; inventory digestion slow.",
        "sentiment": {"supply_trend": "CUT", "demand_trend": "WEAK"},
    },
    {
        "id": "UC4",
        "name": "谣言式新闻",
        "check": "C1",
        "date": "2026-03-01",
        "ddr4": 6.00,
        "dxi": 62000.0,
        "gap": -5.5,
        "horizon_weeks": 4,
        "expected_range": "-2.0% ~ +3.0% (DDR4 spot vs anchor, custom scenario, ~4w)",
        "production_news": "Unverified social posts claim DRAM spot will crash 40% next month.",
        "demand_news": "No verified demand data this week.",
        "sentiment": {"supply_trend": "STABLE", "demand_trend": "NEUTRAL"},
    },
]


def _glm_decision_ablation(prompt: str, max_tokens: int = 1200) -> str:
    if not wo.GLM_API_KEY:
        raise RuntimeError("GLM_API_KEY missing")
    client = ZhipuAI(api_key=wo.GLM_API_KEY)
    last_err: Exception | None = None
    for model_name in (wo.DECISION_MODEL, "glm-4-flash"):
        try:
            rsp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an industrial supply-chain AI. Follow the user's pre-reasoning "
                            "logic baseline strictly; answers must be concise and actionable."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                top_p=0.7,
                max_tokens=max_tokens,
            )
            return rsp.choices[0].message.content.strip()
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"GLM call failed: {last_err!r}")


def _forecast_for_uc(uc: dict) -> wo.TimeLLMForecast:
    return wo.TimeLLMForecast(
        horizon_weeks=int(uc["horizon_weeks"]),
        dxi_trend="neutral_custom",
        expected_change_range_pct=uc["expected_range"],
        rationale=(
            f"[Ablation · custom-mode equivalent] {uc['id']} {uc['name']}; "
            "sentiment labels injected per 用例.md for reproducibility."
        ),
    )


def main(prompt_ver: str) -> None:
    if prompt_ver not in PROMPTS:
        raise SystemExit(f"Unknown --prompt {prompt_ver}; choose: {', '.join(PROMPTS)}")

    template = PROMPTS[prompt_ver]
    ver_upper = prompt_ver.upper()
    out_lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_lines.append(f"# {ver_upper} 提示词自动化测试结果\n")
    out_lines.append(f"- **生成时间（UTC）**: {now}\n")
    out_lines.append("- **模式**: 等价于 Streamlit 自定义模式（`dxi_trend=neutral_custom`，无 Time-LLM）\n")
    out_lines.append(f"- **提示词版本**: {ver_upper}（见 `提示词迭代.md`）\n")
    out_lines.append("- **说明**: `supply_trend` / `demand_trend` 按 `用例.md` **固定注入** `format_prompt_template(..., sentiment=...)`。\n")
    out_lines.append("- **Token 预算**: `max_tokens=1200`（高于 `app.py` 默认 350）。\n")
    out_lines.append("\n---\n")

    for uc in USE_CASES:
        snap = wo.QuantSnapshot(
            date=uc["date"],
            ddr4_spot_price=float(uc["ddr4"]),
            dxi_index=float(uc["dxi"]),
            ddr4_spot_contract_gap_pct=float(uc["gap"]),
        )
        forecast = _forecast_for_uc(uc)
        qualitative = {
            "production_news": uc["production_news"],
            "demand_news": uc["demand_news"],
        }
        sentiment = wo._normalize_sentiment_labels(uc["sentiment"])
        logic_lines = wo.get_evolutionary_logic_lines_from_state(snap, forecast, sentiment)

        out_lines.append(f"\n## {uc['id']} — {uc['name']}\n")
        out_lines.append(f"- **检查项**: {uc['check']}\n")
        out_lines.append("\n### 输入快照（占位符来源）\n")
        out_lines.append("| 字段 | 值 |\n|---|---|\n")
        out_lines.append(f"| ddr4_spot_price | {snap.ddr4_spot_price} |\n")
        out_lines.append(f"| dxi_index | {snap.dxi_index} |\n")
        out_lines.append(f"| gap_pct (raw) | {snap.ddr4_spot_contract_gap_pct} |\n")
        out_lines.append(f"| horizon_weeks | {forecast.horizon_weeks} |\n")
        out_lines.append(f"| expected_change_range_pct | {forecast.expected_change_range_pct} |\n")
        out_lines.append(f"| production_news | {uc['production_news']!r} |\n")
        out_lines.append(f"| demand_news | {uc['demand_news']!r} |\n")
        out_lines.append(f"| supply_trend (fixed) | {sentiment['supply_trend']} |\n")
        out_lines.append(f"| demand_trend (fixed) | {sentiment['demand_trend']} |\n")
        out_lines.append(f"| predict_up | {wo.predict_up_from_forecast(forecast)} |\n")

        out_lines.append(f"\n### 逻辑链条 `logic_lines`（注入 {ver_upper} 前）\n")
        for i, line in enumerate(logic_lines, 1):
            out_lines.append(f"{i}. {line}\n")

        prompt = wo.format_prompt_template(
            template,
            snap,
            forecast,
            qualitative,
            logic_lines=logic_lines,
            sentiment=sentiment,
        )

        out_lines.append(
            f"\n### GLM-4 决策输出（{ver_upper} 填充后 user prompt；`_glm_decision_ablation`，`max_tokens=1200`）\n"
        )
        if wo.DEBUG_MODE:
            out_lines.append("_跳过：workflow_orchestrator.DEBUG_MODE=True_\n")
        elif not wo.GLM_API_KEY:
            out_lines.append("_错误：未设置 GLM_API_KEY，无法调用 API。_\n")
        else:
            try:
                out_lines.append(_glm_decision_ablation(prompt, max_tokens=1200))
                out_lines.append("\n")
            except Exception as exc:
                out_lines.append(f"_调用失败_: `{exc!r}`\n")

        out_lines.append("\n---\n")

    out_name = OUTPUT_FILES[prompt_ver]
    out_path = ROOT / out_name
    out_path.write_text("".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run prompt ablation UC1–UC4.")
    ap.add_argument(
        "--prompt",
        choices=sorted(PROMPTS.keys()),
        required=True,
        help="Prompt template version (v1 / v2 / v3).",
    )
    args = ap.parse_args()
    main(args.prompt)
