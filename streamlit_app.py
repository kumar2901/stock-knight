from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Stock Knight", layout="wide")

st.title("stock-knight")
st.caption("NSE stock ranking pipeline (EOD) — ingest → features/labels → train → rank")

with st.sidebar:
    st.header("Broker")
    broker = st.selectbox("Select broker", ["mock", "yahoo", "zerodha", "upstox", "angelone"])
    creds_file = st.text_input("Credentials file (YAML)", "credentials.yaml")
    st.caption("Create `credentials.yaml` from `credentials.example.yaml` (it is gitignored).")

    st.header("Data")
    data_root = st.text_input("data root", "data")
    start = st.date_input("start date", value=pd.to_datetime("2019-01-01"))
    end = st.date_input("end date", value=pd.to_datetime("2025-12-31"))

    st.header("Symbols")
    symbols_text = st.text_area(
        "Symbols (one per line). For Yahoo: use .NS suffix (e.g., RELIANCE.NS). "
        "For Zerodha: NSE tradingsymbol (e.g., RELIANCE).",
        value="",
        height=150,
    )
    use_symbols = st.checkbox("Use symbols from textbox (recommended for real brokers)", value=True)

    st.header("Training")
    model_kind = st.selectbox("Model kind", ["ridge", "lgbm"])

    st.header("Ranking")
    horizon = st.selectbox("Horizon", ["1d", "7d", "1m", "6m", "12m"])
    asof = st.date_input("Rank as-of date", value=pd.to_datetime("2025-12-30"))


def _run(cmd: str) -> None:
    import subprocess

    st.code(cmd, language="bash")
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if p.returncode != 0:
        st.error(p.stderr or p.stdout)
        raise RuntimeError("Command failed")
    if p.stdout.strip():
        st.text(p.stdout)


symbols_file: Path | None = None
if use_symbols and symbols_text.strip():
    symbols_file = Path(data_root) / "symbols.txt"
    symbols_file.parent.mkdir(parents=True, exist_ok=True)
    symbols_file.write_text(symbols_text.strip() + "\n")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("1) Ingest", use_container_width=True):
        extra = f" --symbols-file {symbols_file}" if symbols_file else ""
        _run(
            f"python -m stock_knight.cli ingest --broker {broker} --data-root {data_root} "
            f"--start {start} --end {end} --creds-file {creds_file}{extra}"
        )
        st.success("Ingest complete.")

with col2:
    if st.button("2) Features + Labels", use_container_width=True):
        ohlcv_path = Path(data_root) / "ohlcv" / f"ohlcv_{start}_{end}.parquet"
        _run(f"python -m stock_knight.cli features-labels --data-root {data_root} --ohlcv-parquet {ohlcv_path}")
        st.success("Features/labels complete.")

with col3:
    if st.button("3) Train", use_container_width=True):
        feat_path = Path(data_root) / "features" / f"features_ohlcv_{start}_{end}.parquet"
        lab_path = Path(data_root) / "labels" / f"labels_ohlcv_{start}_{end}.parquet"
        _run(
            f"python -m stock_knight.cli train --data-root {data_root} "
            f"--features-parquet {feat_path} --labels-parquet {lab_path} --model-kind {model_kind}"
        )
        st.success("Training complete. Models saved in artifacts/models/")

with col4:
    if st.button("4) Rank", use_container_width=True):
        feat_path = Path(data_root) / "features" / f"features_ohlcv_{start}_{end}.parquet"
        model_path = Path("artifacts") / "models" / f"{model_kind}_{horizon}.joblib"
        out_path = Path("reports") / f"ranks_{horizon}_{asof}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run(
            f"python -m stock_knight.cli rank --features-parquet {feat_path} "
            f"--model-path {model_path} --asof {asof} --horizon {horizon} --out {out_path}"
        )
        st.success(f"Wrote {out_path}")

st.divider()

st.subheader("Notes")
st.markdown(
    """
- Credentials can be stored in `credentials.yaml` (recommended) or environment variables as fallback.
- Install extra deps:
  - `pip install -e '.[brokers,ui]'`
"""
)

