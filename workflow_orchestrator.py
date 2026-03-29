"""
# Memory supply-chain decision system (Orchestrator)

Three-stage architecture:
1) Input Layer: structured market variables + unstructured news
2) Processing Layer: Time-LLM forecast + GLM-4 decision (RISEN-style prompt)
3) Output Layer: procurement / inventory guidance + debug logs
"""

from __future__ import annotations

import csv
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from zhipuai import ZhipuAI

_ORCH_ROOT = Path(__file__).resolve().parent
load_dotenv(_ORCH_ROOT / ".env")
load_dotenv()

DEBUG_MODE = False
DATASET_PATH = _ORCH_ROOT / "timellm-dataset/storage/processed/time_llm_dataset_aligned.csv"
GLM_API_KEY = os.getenv("GLM_API_KEY")

# Time-LLM: real inference via timellm_runtime; TIME_LLM_ORCH_DISABLE=1 uses stub forecast
TIME_LLM_ORCH_DISABLE = os.getenv("TIME_LLM_ORCH_DISABLE", "").lower() in ("1", "true", "yes")

# Sentiment (light) and final decision models
SENTIMENT_MODEL = "glm-4.5-air"
DECISION_MODEL = "glm-4"


@dataclass
class QuantSnapshot:
    date: str
    ddr4_spot_price: float
    dxi_index: float
    ddr4_spot_contract_gap_pct: float


@dataclass
class TimeLLMForecast:
    horizon_weeks: int
    dxi_trend: str
    expected_change_range_pct: str
    rationale: str


@dataclass
class DecisionResult:
    risk_level: int
    procurement_ratio: str
    core_logic: List[str]
    inventory_note: str
    moq_note: str
    rendered_instruction: str = ""


# ---------------------------------------------------------------------------
# Stage 1: Multimodal input (Input Layer)
# ---------------------------------------------------------------------------
def load_latest_quant_snapshot(csv_path: Path = DATASET_PATH) -> QuantSnapshot:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Aligned dataset not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Dataset is empty: {csv_path}")

    latest = rows[-1]

    def _to_float(v: Optional[str], field: str) -> float:
        if v is None or str(v).strip() == "":
            raise ValueError(f"Field `{field}` is empty in latest row.")
        return float(v)

    return QuantSnapshot(
        date=str(latest.get("date", "")),
        ddr4_spot_price=_to_float(latest.get("ddr4_spot_price"), "ddr4_spot_price"),
        dxi_index=_to_float(latest.get("dxi_index"), "dxi_index"),
        ddr4_spot_contract_gap_pct=_to_float(
            latest.get("ddr4_spot_contract_gap_pct"), "ddr4_spot_contract_gap_pct"
        ),
    )


def get_qualitative_signals() -> Dict[str, str]:
    """
    Default qualitative inputs:
    - production_news: supply-side headlines
    - demand_news: demand-side headlines
    """
    return {
        "production_news": (
            "Samsung and SK Hynix confirm plans to cut DRAM output by ~15% in Q2."
        ),
        "demand_news": (
            "Handset and server demand remain soft; near-term pull-in momentum is weak."
        ),
    }


# ---------------------------------------------------------------------------
# Stage 2: Dual-engine processing (Processing Layer)
# ---------------------------------------------------------------------------
def forecast_custom_static(
    snapshot: QuantSnapshot,
    weeks: int = 8,
    *,
    expected_change_range_pct: Optional[str] = None,
) -> TimeLLMForecast:
    """
    Custom scenario mode: no Time-LLM; QuantSnapshot is static context for GLM only.
    ``dxi_trend`` stays ``neutral_custom`` (no built-in bullish keyword); optional
    ``expected_change_range_pct`` fills the RISEN placeholder in the panel.
    """
    weeks = max(4, min(16, int(weeks)))
    ec_raw = (expected_change_range_pct or "").strip()
    ec = (
        ec_raw
        if ec_raw
        else "N/A (custom mode — set expected_change_range_pct in the panel above)"
    )
    rationale_extra = (
        f" User-stated outlook band: {ec_raw}."
        if ec_raw
        else " No explicit % range set; use quantitative sliders for levels."
    )
    return TimeLLMForecast(
        horizon_weeks=weeks,
        dxi_trend="neutral_custom",
        expected_change_range_pct=ec,
        rationale=(
            f"[Custom scenario · Time-LLM off] DDR4 spot={snapshot.ddr4_spot_price:.4f}, "
            f"DXI={snapshot.dxi_index:.2f}, Gap%={snapshot.ddr4_spot_contract_gap_pct:.4f}. "
            "Values come from UI sliders and serve as LLM context only."
            + rationale_extra
        ),
    )


def _timellm_simulation_stub(snapshot: QuantSnapshot, weeks: int = 8) -> TimeLLMForecast:
    """Stub when weights missing, OOM, or TIME_LLM_ORCH_DISABLE."""
    weeks = max(4, min(8, int(weeks)))
    return TimeLLMForecast(
        horizon_weeks=weeks,
        dxi_trend="bullish_rebound",
        expected_change_range_pct="5% - 8%",
        rationale=(
            "[Stub] Short-term tape shows basing; spot/contract dislocation is expected to narrow. "
            "Placeholder path assumes a mild 8-week upside band (Time-LLM checkpoint not loaded)."
        ),
    )


def timellm_simulation_engine(snapshot: QuantSnapshot, weeks: int = 8) -> TimeLLMForecast:
    """
    Time-LLM forecast: `timellm_runtime.try_real_timellm_forecast` on the last window of
    `time_llm_data_cleaned.csv` (same stack as Streamlit time-series mode / plot script).

    Independent of sidebar QuantSnapshot when called from CLI; sliders only affect GLM downstream in app.

    Env:
      TIME_LLM_ORCH_DISABLE=1 — force stub
      TIME_LLM_MODEL_PATH — required local Qwen dir or HF id (set in `.env` or export)
      TIME_LLM_CHECKPOINT — default ``<repo>/checkpoints/timellm_best.pth``
      TIME_LLM_DATA_PATH — default time_llm_data_cleaned.csv
      TIME_LLM_ROOT — default ``<repo>/timellm-dataset/storage/processed``
    """
    if TIME_LLM_ORCH_DISABLE:
        return _timellm_simulation_stub(snapshot, weeks)

    try:
        from timellm_runtime import try_real_timellm_forecast

        return try_real_timellm_forecast(weeks=weeks)
    except Exception as exc:
        if os.getenv("TIME_LLM_ORCH_STRICT", "").lower() in ("1", "true", "yes"):
            raise
        warnings.warn(
            f"[Time-LLM] inference failed, using stub forecast: {exc}",
            stacklevel=2,
        )
        return _timellm_simulation_stub(snapshot, weeks)


def _predict_up_from_forecast(forecast: TimeLLMForecast) -> bool:
    forecast_text = f"{forecast.dxi_trend} {forecast.expected_change_range_pct}".lower()
    return ("bullish" in forecast_text) or ("upside" in forecast_text)


def predict_up_from_forecast(forecast: TimeLLMForecast) -> bool:
    """Whether forecast implies upside (UI / routing)."""
    return _predict_up_from_forecast(forecast)


def _sentiment_fallback(production_news: str, demand_news: str) -> Dict[str, str]:
    """Rule-based fallback when sentiment API fails (English + legacy Chinese keywords)."""
    prod = production_news or ""
    dem = demand_news or ""
    pl = prod.lower()
    dl = dem.lower()
    supply_trend = "STABLE"
    if any(
        k in pl
        for k in (
            "cut output",
            "cut production",
            "reduce output",
            "production cut",
            "curtail",
            "trim capacity",
        )
    ) or any(k in prod for k in ("减产", "削减", "收缩", "削减产出", "削减 DRAM")):
        supply_trend = "CUT"
    if any(
        k in pl
        for k in ("ramp", "expand capacity", "add capacity", "increase output", "capacity add")
    ) or any(k in prod for k in ("增产", "扩产", "复产", "扩张", "产能释放")):
        supply_trend = "INCREASE"
    demand_trend = "NEUTRAL"
    if any(k in dl for k in ("weak", "soft", "sluggish", "slowdown", "decline")) or any(
        k in dem for k in ("疲软", "偏弱", "下滑", "下降")
    ):
        demand_trend = "WEAK"
    if any(k in dl for k in ("strong", "robust", "recovery", "firm demand")) or any(
        k in dem for k in ("强劲", "旺盛", "回暖", "上升")
    ):
        demand_trend = "STRONG"
    return {"supply_trend": supply_trend, "demand_trend": demand_trend}


def _normalize_sentiment_labels(raw: Dict[str, str]) -> Dict[str, str]:
    s = str(raw.get("supply_trend", "STABLE")).upper().strip()
    d = str(raw.get("demand_trend", "NEUTRAL")).upper().strip()
    if s not in {"CUT", "INCREASE", "STABLE"}:
        s = "STABLE"
    if d not in {"WEAK", "STRONG", "NEUTRAL"}:
        d = "NEUTRAL"
    return {"supply_trend": s, "demand_trend": d}


def parse_market_sentiment(production_news: str, demand_news: str) -> Dict[str, str]:
    """
    Sentiment layer: GLM-4.5-Air maps unstructured news to discrete labels.
    JSON only: supply_trend ∈ {CUT, INCREASE, STABLE}; demand_trend ∈ {WEAK, STRONG, NEUTRAL}.
    """
    if not GLM_API_KEY:
        return _normalize_sentiment_labels(_sentiment_fallback(production_news, demand_news))

    system = (
        "You are a strict JSON classifier for semiconductor supply-chain text. "
        "Output ONLY one JSON object, no markdown, no explanation."
    )
    user = (
        "Analyze the following texts (any language).\n\n"
        f"production_news: {production_news!r}\n"
        f"demand_news: {demand_news!r}\n\n"
        "Return JSON with exactly these keys:\n"
        '{"supply_trend":"CUT|INCREASE|STABLE","demand_trend":"WEAK|STRONG|NEUTRAL"}\n'
        "Rules:\n"
        "- CUT: output cuts, fab curtailment, planned reduction in supply\n"
        "- INCREASE: ramp-ups, new capacity, higher planned output\n"
        "- STABLE: neutral supply / no clear expansion or contraction\n"
        "- WEAK: soft demand, downside, sluggish end markets\n"
        "- STRONG: firm demand, recovery, strong pull-in\n"
        "- NEUTRAL: demand unclear or insufficient information\n"
    )
    try:
        client = ZhipuAI(api_key=GLM_API_KEY)
        text = ""
        for model_name in (SENTIMENT_MODEL, "glm-4-air", "glm-4-flash"):
            try:
                rsp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=0.5,
                    max_tokens=128,
                )
                text = rsp.choices[0].message.content.strip()
                break
            except Exception:
                continue
        if not text:
            raise RuntimeError("sentiment models unavailable")
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not a dict")
        return _normalize_sentiment_labels(
            {
                "supply_trend": str(data.get("supply_trend", "STABLE")),
                "demand_trend": str(data.get("demand_trend", "NEUTRAL")),
            }
        )
    except Exception:
        return _normalize_sentiment_labels(_sentiment_fallback(production_news, demand_news))


# Default user prompt for GLM-4 decision (Version 3 from prompt ablation; placeholders via format_prompt_template).
DEFAULT_RISEN_PROMPT_TEMPLATE = """Role: Industrial memory procurement advisor (conservative, audit-friendly).

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

Max ~250 words. Chinese or English.
"""


def build_risen_prompt(snapshot: QuantSnapshot, forecast: TimeLLMForecast, qualitative: Dict[str, str]) -> str:
    return format_prompt_template(
        DEFAULT_RISEN_PROMPT_TEMPLATE, snapshot, forecast, qualitative
    )


def build_base_prompt(snapshot: QuantSnapshot, forecast: TimeLLMForecast, qualitative: Dict[str, str]) -> str:
    """Short baseline prompt for A/B tests."""
    return (
        "You are a memory procurement advisor. Given the data below, give procurement guidance "
        "(risk 1-5, suggested ratio, one-line rationale), max ~200 words, bullets OK.\n\n"
        f"DDR4 spot: {snapshot.ddr4_spot_price}\n"
        f"DXI: {snapshot.dxi_index}\n"
        f"Gap%: {snapshot.ddr4_spot_contract_gap_pct:.4f}\n"
        f"Supply news: {qualitative.get('production_news', '')}\n"
        f"Demand news: {qualitative.get('demand_news', '')}\n"
        f"Model hint: over {forecast.horizon_weeks} weeks, outlook {forecast.expected_change_range_pct}.\n"
    )


def format_prompt_template(
    template: str,
    snapshot: QuantSnapshot,
    forecast: TimeLLMForecast,
    qualitative: Dict[str, str],
    logic_lines: Optional[List[str]] = None,
    sentiment: Optional[Dict[str, str]] = None,
) -> str:
    """Fill template placeholders from snapshot, forecast, and qualitative inputs."""
    gap_pct = f"{snapshot.ddr4_spot_contract_gap_pct:.4f}"
    prod = qualitative.get("production_news", "") or ""
    dem = qualitative.get("demand_news", "") or ""
    if sentiment is None:
        sentiment = parse_market_sentiment(prod, dem)
    sentiment = _normalize_sentiment_labels(sentiment)
    if logic_lines is None:
        logic_lines = get_evolutionary_logic_lines_from_state(snapshot, forecast, sentiment)
    logic_text = "\n".join(f"- {line}" for line in logic_lines)
    return template.format(
        gap_pct=gap_pct,
        ddr4_spot_price=snapshot.ddr4_spot_price,
        dxi_index=snapshot.dxi_index,
        horizon_weeks=forecast.horizon_weeks,
        expected_change_range_pct=forecast.expected_change_range_pct,
        production_news=prod,
        demand_news=dem,
        forecast_rationale=forecast.rationale,
        supply_trend=sentiment["supply_trend"],
        demand_trend=sentiment["demand_trend"],
        logic_lines=logic_text,
    )


def get_evolutionary_logic_lines(
    gap_pct: float,
    dxi_index: float,
    predict_up: bool,
    supply_trend: str,
    demand_trend: str,
) -> List[str]:
    """
    Logic router: combine GLM sentiment tags and numeric thresholds into five logic_lines
    (conflict/resonance + Step-1/2/3).
    """
    st = str(supply_trend).upper().strip()
    dt = str(demand_trend).upper().strip()
    logic_lines: List[str] = []

    # A. Conflict / resonance (predict_up vs demand_trend)
    if predict_up and dt == "WEAK":
        logic_lines.append(
            "[CONFLICT] Forecast skews bullish while demand is weak—use risk-hedge framing."
        )
    else:
        logic_lines.append(
            "[RESONANCE] Macro forecast aligns with micro signals; higher decision clarity."
        )

    # B. Step-1 supply parser
    if st == "CUT" and gap_pct < -15:
        logic_lines.append(
            "Step-1 Parser: Cuts are confirmed, but deep spot discount shows feeble confidence; "
            "cut support not yet priced in."
        )
    elif st == "CUT" and gap_pct >= -15:
        logic_lines.append(
            "Step-1 Parser: Supply contraction is partially validated in spreads; basing support credible."
        )
    elif st == "INCREASE":
        logic_lines.append(
            "Step-1 Parser: Capacity adds are explicit; prices face downside pressure from higher supply."
        )
    else:
        logic_lines.append(
            "Step-1 Parser: Supply baseline; price action is mostly demand-driven."
        )

    # C. Step-2 price momentum (DXI level)
    if dxi_index > 75000:
        logic_lines.append(
            "Step-2 Reasoner: DXI in a 75000+ stretch—watch for pullback risk in a stretched tape."
        )
    elif dxi_index < 40000:
        logic_lines.append(
            "Step-2 Reasoner: DXI in a low band—room for technical rebound / mean reversion."
        )
    else:
        logic_lines.append(
            "Step-2 Reasoner: DXI mid-range (40000–75000); track gap repair before sizing risk."
        )

    # D. Step-3 demand guardrail
    if dt == "WEAK":
        logic_lines.append(
            "Step-3 Guardrail: Weak demand keeps tension high; favor small tranches and MOQ-sized buys."
        )
    else:
        logic_lines.append(
            "Step-3 Guardrail: Chain closes; scale tactical or strategic buys to inventory gaps."
        )

    return logic_lines


def get_evolutionary_logic_lines_from_state(
    snapshot: QuantSnapshot,
    forecast: TimeLLMForecast,
    sentiment: Dict[str, str],
) -> List[str]:
    """Build logic_lines from snapshot, forecast, and parsed sentiment."""
    sentiment = _normalize_sentiment_labels(sentiment)
    return get_evolutionary_logic_lines(
        float(snapshot.ddr4_spot_contract_gap_pct),
        float(snapshot.dxi_index),
        _predict_up_from_forecast(forecast),
        sentiment["supply_trend"],
        sentiment["demand_trend"],
    )


def log_evolutionary_logic(
    snapshot: QuantSnapshot,
    forecast: TimeLLMForecast,
    qualitative: Dict[str, str],
    debug_mode: bool = DEBUG_MODE,
) -> List[str]:
    """Debug helper: sentiment + router -> EvoLog lines."""
    sentiment = parse_market_sentiment(
        qualitative.get("production_news", "") or "",
        qualitative.get("demand_news", "") or "",
    )
    logs = get_evolutionary_logic_lines_from_state(snapshot, forecast, sentiment)
    if debug_mode:
        for line in logs:
            print(f"[DEBUG][EvoLog] {line}")
    return logs


def _fallback_decision(snapshot: QuantSnapshot, forecast: TimeLLMForecast, qualitative: Dict[str, str]) -> DecisionResult:
    dn = (qualitative.get("demand_news", "") or "").lower()
    demand_weak = any(
        k in dn for k in ("weak", "soft", "sluggish", "疲软", "偏弱", "下滑")
    )
    gap_abs = abs(snapshot.ddr4_spot_contract_gap_pct)
    if demand_weak and gap_abs > 20:
        return DecisionResult(
            risk_level=4,
            procurement_ratio="30% - 40%",
            core_logic=[
                "Supply cuts support mid-term pricing, but weak demand caps near-term repricing.",
                "Gap% still deeply negative—price discovery not fully healed.",
                "Stagger buys to limit drawdown; keep dry powder for follow-on tranches.",
            ],
            inventory_note="Target 4–6 weeks coverage; avoid overstock and cash tie-up.",
            moq_note="Split orders at MOQ; secure strategic SKUs first.",
            rendered_instruction=(
                "- Risk level: 4\n- Procurement: 30%-40%\n- Rationale: supply tight but demand soft; tranche buys."
            ),
        )
    return DecisionResult(
        risk_level=3,
        procurement_ratio="45% - 55%",
        core_logic=[
            "Supply cuts and constructive model view open a tactical restock window.",
            "Dislocation is manageable; path to spread repair looks orderly.",
        ],
        inventory_note="Lift coverage toward 6–8 weeks; track demand slope.",
        moq_note="Within MOQ rules, protect continuity on high-runners.",
        rendered_instruction=(
            "- Risk level: 3\n- Procurement: 45%-55%\n- Rationale: balanced repair; steady restock."
        ),
    )


def glm4_decision_engine(snapshot: QuantSnapshot, forecast: TimeLLMForecast, qualitative: Dict[str, str]) -> DecisionResult:
    """
    GLM-4 decision engine (RISEN prompt).
    With GLM_API_KEY calls the API; on failure falls back to rule engine if not strict.
    """
    prompt = build_risen_prompt(snapshot, forecast, qualitative)
    if DEBUG_MODE:
        return _fallback_decision(snapshot, forecast, qualitative)

    if not GLM_API_KEY:
        raise RuntimeError("GLM_API_KEY is required when DEBUG_MODE=False")

    try:
        client = ZhipuAI(api_key=GLM_API_KEY)
        content = ""
        last_err: Optional[Exception] = None
        for model_name in (DECISION_MODEL, "glm-4-flash"):
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
                    max_tokens=350,
                )
                content = rsp.choices[0].message.content.strip()
                break
            except Exception as exc:  # pragma: no cover
                last_err = exc
                continue
        if not content:
            raise RuntimeError(f"Decision model call failed: {last_err!r}")
        return DecisionResult(
            risk_level=0,
            procurement_ratio="(see rendered_instruction)",
            core_logic=["(see rendered_instruction)"],
            inventory_note="(see rendered_instruction)",
            moq_note="(see rendered_instruction)",
            rendered_instruction=content,
        )
    except Exception as exc:
        if DEBUG_MODE:
            print(f"[DEBUG] GLM API failed, fallback to rule engine: {exc}")
        return _fallback_decision(snapshot, forecast, qualitative)


def glm4_decision_with_prompt(prompt: str) -> DecisionResult:
    """
    Call the decision model (default GLM-4) with an arbitrary user prompt (Streamlit template).
    Raises if DEBUG_MODE=True (no API).
    """
    if DEBUG_MODE:
        raise RuntimeError("glm4_decision_with_prompt requires DEBUG_MODE=False.")

    if not GLM_API_KEY:
        raise RuntimeError("GLM_API_KEY is required when DEBUG_MODE=False")

    client = ZhipuAI(api_key=GLM_API_KEY)
    last_err: Optional[Exception] = None
    content = ""
    for model_name in (DECISION_MODEL, "glm-4-flash"):
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
                max_tokens=350,
            )
            content = rsp.choices[0].message.content.strip()
            break
        except Exception as exc:  # pragma: no cover
            last_err = exc
            continue
    if not content:
        raise RuntimeError(f"Decision model call failed: {last_err!r}")

    return DecisionResult(
        risk_level=0,
        procurement_ratio="(see rendered_instruction)",
        core_logic=["(see rendered_instruction)"],
        inventory_note="(see rendered_instruction)",
        moq_note="(see rendered_instruction)",
        rendered_instruction=content,
    )


# ---------------------------------------------------------------------------
# Stage 3: Decision output (Output Layer)
# ---------------------------------------------------------------------------
def generate_procurement_instruction(
    snapshot: QuantSnapshot, forecast: TimeLLMForecast, decision: DecisionResult
) -> Dict[str, object]:
    return {
        "title": "Memory procurement & inventory dispatch brief",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "market_snapshot": {
            "date": snapshot.date,
            "ddr4_spot_price": snapshot.ddr4_spot_price,
            "dxi_index": snapshot.dxi_index,
            "ddr4_spot_contract_gap_pct": snapshot.ddr4_spot_contract_gap_pct,
        },
        "forecast_summary": {
            "horizon_weeks": forecast.horizon_weeks,
            "trend": forecast.dxi_trend,
            "expected_change_range_pct": forecast.expected_change_range_pct,
        },
        "decision": {
            "risk_level_1_to_5": decision.risk_level,
            "procurement_ratio": decision.procurement_ratio,
            "core_logic": decision.core_logic,
            "inventory_note": decision.inventory_note,
            "moq_note": decision.moq_note,
        },
        "decision_markdown": decision.rendered_instruction,
    }


def run_orchestrator() -> Dict[str, object]:
    snapshot = load_latest_quant_snapshot(DATASET_PATH)
    qualitative = get_qualitative_signals()
    forecast = timellm_simulation_engine(snapshot, weeks=8)
    sentiment = parse_market_sentiment(
        qualitative.get("production_news", "") or "",
        qualitative.get("demand_news", "") or "",
    )
    logic_lines = get_evolutionary_logic_lines_from_state(snapshot, forecast, sentiment)
    if DEBUG_MODE:
        for line in logic_lines:
            print(f"[DEBUG][EvoLog] {line}")
    decision = glm4_decision_engine(snapshot, forecast, qualitative)
    return generate_procurement_instruction(snapshot, forecast, decision)


if __name__ == "__main__":
    instruction = run_orchestrator()
    print(json.dumps(instruction, ensure_ascii=False, indent=2))
    if DEBUG_MODE and not GLM_API_KEY:
        print("[DEBUG] GLM_API_KEY unset; using offline rule engine.")

