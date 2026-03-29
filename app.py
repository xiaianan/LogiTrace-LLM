"""
Memory supply-chain decision system — Streamlit panel (RISEN template + single GLM call).
Run: streamlit run app.py
Deps: pip install streamlit python-dotenv zhipuai pandas
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")
load_dotenv()

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit", file=sys.stderr)
    sys.exit(1)

import numpy as np
import pandas as pd

import workflow_orchestrator as wo  # noqa: E402


def _load_csv_slider_defaults():
    """Slider defaults from aligned CSV latest row; fall back to constants."""
    try:
        snap = wo.load_latest_quant_snapshot(wo.DATASET_PATH)
        return {
            "date": snap.date,
            "ddr4_spot_price": float(snap.ddr4_spot_price),
            "dxi_index": float(snap.dxi_index),
            "ddr4_spot_contract_gap_pct": float(snap.ddr4_spot_contract_gap_pct),
        }
    except Exception:
        return {
            "date": "simulated",
            "ddr4_spot_price": 6.18,
            "dxi_index": 51032.0,
            "ddr4_spot_contract_gap_pct": -12.59,
        }


def _snapshot_from_cleaned_idx(df: pd.DataFrame, row_idx: int) -> wo.QuantSnapshot:
    row = df.iloc[int(row_idx)]
    return wo.QuantSnapshot(
        date=str(row["date"]),
        ddr4_spot_price=float(row["ddr4_spot_price"]),
        dxi_index=float(row["dxi_index"]),
        ddr4_spot_contract_gap_pct=float(row["ddr4_spot_contract_gap_pct"]),
    )


def _forecast_table_pred_only(fd) -> pd.DataFrame:
    """ForecastAtDate -> wide table with prediction columns only (no ground truth)."""
    h, _ = fd.pred.shape
    data: dict = {
        "step": list(range(1, h + 1)),
        "calendar_date": fd.horizon_dates,
    }
    for j, name in enumerate(fd.column_names):
        data[f"pred_{name}"] = fd.pred[:, j].astype(float)
    if "dxi_index_log" in fd.column_names:
        li = fd.column_names.index("dxi_index_log")
        data["pred_implied_dxi"] = np.exp(fd.pred[:, li])
    return pd.DataFrame(data)


def main():
    st.set_page_config(
        page_title="Memory supply chain · Panel",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Memory supply-chain decision system · Interactive panel")
    if "risen_editor" not in st.session_state:
        st.session_state.risen_editor = wo.DEFAULT_RISEN_PROMPT_TEMPLATE

    if wo.DEBUG_MODE:
        st.warning(
            "`workflow_orchestrator.DEBUG_MODE=True` — GLM API is not called. "
            "Set `DEBUG_MODE = False` in `workflow_orchestrator.py` for online decisions."
        )

    defaults = _load_csv_slider_defaults()

    st.sidebar.header("RISEN prompt template")
    if st.sidebar.button("Reset to default RISEN template"):
        st.session_state.risen_editor = wo.DEFAULT_RISEN_PROMPT_TEMPLATE
        st.rerun()

    template_body = st.sidebar.text_area(
        "Custom prompt template",
        height=520,
        key="risen_editor",
        help="After editing, click Run decision to fill placeholders and call GLM-4.",
    )

    st.subheader("1. Multimodal input (Input Layer)")

    ts_mode = st.toggle(
        "Time-series mode (Time-LLM)",
        value=False,
        help="On: pick a history cutoff on the cleaned series and run Time-LLM. "
        "Off: DDR4 / DXI / Gap% from sliders only; no Time-LLM — static scenario for GLM.",
    )

    qualitative = {}
    forecast = None
    snapshot = None
    fd_bundle = None
    cleaned_df_for_snap = None

    if ts_mode:
        try:
            import timellm_runtime as tr

            slots = tr.get_timeline_slots_last_month()
            if not slots:
                st.error(
                    "Cannot build timeline: data too short or path wrong. "
                    "Check `time_llm_data_cleaned.csv` and TIME_LLM_* env vars."
                )
                st.stop()

            labels = [s[1] for s in slots]
            idx_list = [s[0] for s in slots]

            choice_label = st.select_slider(
                "History cutoff (anchor)",
                options=labels,
                value=labels[-1],
                help="Encoder last row aligns to this date; forecast is the next pred_len steps.",
            )
            end_idx = idx_list[labels.index(choice_label)]

            args_probe = tr._prepare_args()
            cleaned_df_for_snap, _, _ = tr._load_series_for_timellm(args_probe)
            snapshot = _snapshot_from_cleaned_idx(cleaned_df_for_snap, end_idx)

            if wo.TIME_LLM_ORCH_DISABLE:
                st.warning(
                    "`TIME_LLM_ORCH_DISABLE=1` — Time-LLM cannot load in time-series mode. "
                    "Unset to enable GPU inference."
                )
                forecast = wo.forecast_custom_static(snapshot)
            else:
                with st.spinner("Time-LLM inferencing… (first load of Qwen can be slow)"):
                    try:
                        forecast, fd_bundle = tr.orchestrator_forecast_at_end_idx(end_idx, weeks=8)
                        extra = (
                            f" anchor_date={fd_bundle.anchor_date}, global_end_idx={fd_bundle.global_end_idx}."
                        )
                        forecast = wo.TimeLLMForecast(
                            horizon_weeks=forecast.horizon_weeks,
                            dxi_trend=forecast.dxi_trend,
                            expected_change_range_pct=forecast.expected_change_range_pct,
                            rationale=forecast.rationale + extra,
                        )
                    except Exception as e:
                        st.error(f"Time-LLM inference failed: {e}")
                        st.info("Check `TIME_LLM_MODEL_PATH`, GPU memory, and checkpoint path.")
                        st.stop()

            st.success(
                f"Anchor **{snapshot.date}** | spot {snapshot.ddr4_spot_price:.4f} | "
                f"DXI {snapshot.dxi_index:.1f} | Gap% {snapshot.ddr4_spot_contract_gap_pct:.4f}"
            )

            if fd_bundle is not None:
                st.markdown("**Next-horizon step forecast**")
                st.dataframe(_forecast_table_pred_only(fd_bundle), use_container_width=True)

        except Exception as e:
            st.exception(e)
            st.stop()

        c2 = st.container()
        with c2:
            st.markdown("**Qualitative**")
            qualitative["production_news"] = st.text_area(
                "Supply — production_news",
                value=wo.get_qualitative_signals()["production_news"],
                height=100,
                key="prod_ts",
            )
            qualitative["demand_news"] = st.text_area(
                "Demand — demand_news",
                value=wo.get_qualitative_signals()["demand_news"],
                height=100,
                key="dem_ts",
            )

    else:
        st.info(
            "**Custom scenario**: slider values are **not** sent to Time-LLM; "
            "they are static quantitative context for GLM only."
        )
        st.markdown("**Template placeholders (custom mode)**")
        st.caption(
            "Sliders fill `{horizon_weeks}` and `{expected_change_range_pct}` in the RISEN template "
            "(Time-LLM off; % band is vs current DDR4 spot level as a scenario label)."
        )
        custom_horizon_weeks = st.select_slider(
            "horizon_weeks",
            options=[4, 8, 12, 16],
            value=8,
            key="custom_horizon_weeks",
            help="Forecast horizon in weeks for {horizon_weeks} (four presets).",
        )
        c_rng_lo, c_rng_hi = st.columns(2)
        with c_rng_lo:
            custom_pct_lo = st.slider(
                "expected_change_range — lower (% vs spot)",
                min_value=-50.0,
                max_value=50.0,
                value=-5.0,
                step=0.5,
                key="custom_pct_lo",
                help="Lower end of % band for {expected_change_range_pct}.",
            )
        with c_rng_hi:
            custom_pct_hi = st.slider(
                "expected_change_range — upper (% vs spot)",
                min_value=-50.0,
                max_value=50.0,
                value=8.0,
                step=0.5,
                key="custom_pct_hi",
                help="Upper end of % band for {expected_change_range_pct}.",
            )
        lo, hi = min(custom_pct_lo, custom_pct_hi), max(custom_pct_lo, custom_pct_hi)
        custom_expected_range = (
            f"{lo:.1f}% ~ {hi:.1f}% (DDR4 spot vs anchor, custom scenario, "
            f"~{int(custom_horizon_weeks)}w)"
        )
        st.caption(f"**effective** `expected_change_range_pct` → `{custom_expected_range}`")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Quantitative (sliders)**")
            ddr4 = st.slider(
                "DDR4 spot price",
                min_value=0.5,
                max_value=30.0,
                value=float(defaults["ddr4_spot_price"]),
                step=0.01,
                key="ddr4_c",
            )
            dxi = st.slider(
                "DXI index",
                min_value=5000.0,
                max_value=120000.0,
                value=float(defaults["dxi_index"]),
                step=1.0,
                key="dxi_c",
            )
            gap_pct = st.slider(
                "Gap% (spot vs contract)",
                min_value=-80.0,
                max_value=30.0,
                value=float(defaults["ddr4_spot_contract_gap_pct"]),
                step=0.01,
                key="gap_c",
            )
            st.caption(f"Reference date (aligned CSV): `{defaults['date']}`")

        with c2:
            st.markdown("**Qualitative**")
            qualitative["production_news"] = st.text_area(
                "Supply — production_news",
                value=wo.get_qualitative_signals()["production_news"],
                height=100,
                key="prod_c",
            )
            qualitative["demand_news"] = st.text_area(
                "Demand — demand_news",
                value=wo.get_qualitative_signals()["demand_news"],
                height=100,
                key="dem_c",
            )

        snapshot = wo.QuantSnapshot(
            date=str(defaults.get("date") or "simulated"),
            ddr4_spot_price=ddr4,
            dxi_index=dxi,
            ddr4_spot_contract_gap_pct=gap_pct,
        )
        forecast = wo.forecast_custom_static(
            snapshot,
            weeks=int(custom_horizon_weeks),
            expected_change_range_pct=custom_expected_range,
        )

    assert snapshot is not None and forecast is not None
    qualitative = {
        "production_news": qualitative["production_news"].strip(),
        "demand_news": qualitative["demand_news"].strip(),
    }

    gap_pct_val = float(snapshot.ddr4_spot_contract_gap_pct)
    if gap_pct_val < -15:
        gap_status = "SEVERE_DISCOUNT"
    elif -15 <= gap_pct_val <= 0:
        gap_status = "NORMAL_GAP"
    else:
        gap_status = "OVERHEATED"

    dxi_val = float(snapshot.dxi_index)
    if dxi_val < 40000:
        dxi_band = "LOW (<40000)"
    elif dxi_val <= 75000:
        dxi_band = "MID (40000–75000)"
    else:
        dxi_band = "HIGH (>75000)"

    predict_up = wo.predict_up_from_forecast(forecast)

    st.sidebar.divider()
    st.sidebar.subheader("State panel (State Mapping)")
    st.sidebar.markdown(
        "| Field | Value |\n"
        f"|---|---|\n"
        f"| Mode | {'Time-series (Time-LLM)' if ts_mode else 'Custom (no Time-LLM)'} |\n"
        f"| Gap_Status | {gap_status} (Gap%={gap_pct_val:.4f}%) |\n"
        f"| DXI band (routing) | {dxi_band} (DXI={dxi_val:.1f}) |\n"
        f"| Time-LLM predict_up | {predict_up} |\n"
    )
    st.sidebar.caption(
        "supply_trend / demand_trend are parsed by GLM-4.5-Air when you click Run decision."
    )

    with st.expander("Time-LLM summary (fed into prompt)", expanded=not ts_mode):
        st.markdown(f"**horizon_weeks**={forecast.horizon_weeks}")
        st.markdown(f"**dxi_trend**={forecast.dxi_trend}")
        st.markdown(f"**expected_change_range_pct**={forecast.expected_change_range_pct}")
        st.markdown(forecast.rationale)

    st.subheader("2. Processing engine (Processing Layer)")
    if st.button("Run decision", type="primary"):
        if wo.DEBUG_MODE:
            st.error("DEBUG_MODE is on — GLM cannot be called. Turn it off and retry.")
            return

        if not wo.GLM_API_KEY:
            st.error("GLM_API_KEY not found. Set it in `.env` or the environment.")
            return

        with st.spinner("GLM-4.5-Air parsing sentiment…"):
            sentiment = wo.parse_market_sentiment(
                qualitative.get("production_news", "") or "",
                qualitative.get("demand_news", "") or "",
            )
        with st.expander("Sentiment layer (GLM-4.5-Air)", expanded=True):
            st.json(sentiment)

        evo_lines = wo.get_evolutionary_logic_lines_from_state(snapshot, forecast, sentiment)
        st.markdown("**Logic log / Evolutionary Logic Log**")
        for line in evo_lines:
            st.markdown(f"- {line}")

        try:
            filled_prompt = wo.format_prompt_template(
                template_body,
                snapshot,
                forecast,
                qualitative,
                logic_lines=evo_lines,
                sentiment=sentiment,
            )
            decision_main = wo.glm4_decision_with_prompt(filled_prompt)
        except KeyError as e:
            st.error(f"Template placeholder mismatch — check brace variable names: {e}")
            return
        except Exception as e:
            st.error(f"GLM call failed: {e}")
            return

        st.subheader("3. Decision output (Output Layer)")
        st.markdown("**GLM-4 · Decision (Markdown; follow logic_lines tone)**")
        st.markdown(decision_main.rendered_instruction)


if __name__ == "__main__":
    main()
