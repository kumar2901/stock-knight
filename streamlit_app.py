"""
Stock Knight — NSE stock ranking dashboard.

Tabs:
  1. Rankings  — latest top-K stocks by predicted return, with live LTP
  2. Pipeline  — ingest / features / train / rank
  3. Trading   — Angel One / Zerodha order execution
"""
from __future__ import annotations

import json
import shlex
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Stock Knight ♞",
    page_icon="♞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — persistent settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chess-knight.png", width=60)
    st.title("Stock Knight ♞")
    st.caption("NSE stock ranking & trading")

    st.divider()
    st.header("Broker")
    broker = st.selectbox(
        "Select broker",
        ["yahoo", "angelone", "upstox", "zerodha", "mock"],
        help="Yahoo = free, no login. Angel One / Upstox = free with demat account.",
    )
    creds_file = st.text_input("Credentials file", "credentials.yaml")

    st.divider()
    st.header("Data")
    data_root = st.text_input("Data root folder", "data")
    today = date.today()
    start = st.date_input("History start", value=date(2019, 1, 1))
    end = st.date_input("History end", value=today)

    st.divider()
    st.header("Symbols")
    symbols_file_path = st.text_input("Symbols file", "symbols.txt")
    symbols_text = st.text_area("Or paste symbols (one per line)", value="", height=120)
    use_text_symbols = st.checkbox("Use pasted symbols above", value=False)

    st.divider()
    st.header("Model")
    model_kind = st.selectbox("Model", ["ridge", "lgbm"])
    train_target = st.selectbox("Training target", ["excess", "raw"])

    # Benchmark index options per broker — only verified/eligible tickers listed.
    # Yahoo: tested live; ^CNX500, ^NIFTYBANK, ^CNXMIDCAP are broken on Yahoo Finance.
    # Zerodha: Kite Connect instrument tradingsymbol for NSE indices (exchange=NSE).
    # Angel One: SmartAPI exchange=NSE instrument names for indices.
    # Upstox: NSE_INDEX|<name> instrument key format.
    _BENCHMARKS: dict[str, list[tuple[str, str]]] = {
        "yahoo": [
            # Verified working as of 2026-05 via yfinance
            ("Nifty 50  (^NSEI)",         "^NSEI"),
            ("Nifty 100 (^CNX100)",        "^CNX100"),
            ("Nifty 200 (^CNX200)",        "^CNX200"),
            ("Nifty Midcap (^NSMIDCP)",    "^NSMIDCP"),
            ("Sensex BSE (^BSESN)",        "^BSESN"),
            ("None — no benchmark",        ""),
        ],
        "zerodha": [
            # Kite Connect: exchange=NSE, instrument_type=INDEX
            ("Nifty 50",           "NIFTY 50"),
            ("Nifty Bank",         "NIFTY BANK"),
            ("Nifty Midcap 50",    "NIFTY MIDCAP 50"),
            ("Nifty 100",          "NIFTY 100"),
            ("Nifty 200",          "NIFTY 200"),
            ("None — no benchmark", ""),
        ],
        "angelone": [
            # SmartAPI: exchange=NSE, symboltoken for indices
            ("Nifty 50",           "NIFTY"),
            ("Nifty Bank",         "BANKNIFTY"),
            ("Nifty Midcap 50",    "MIDCPNIFTY"),
            ("Nifty 100",          "NIFTY100"),
            ("None — no benchmark", ""),
        ],
        "upstox": [
            # Upstox instrument key format: NSE_INDEX|<exchange_name>
            ("Nifty 50",           "NSE_INDEX|Nifty 50"),
            ("Nifty Bank",         "NSE_INDEX|Nifty Bank"),
            ("Nifty Midcap 50",    "NSE_INDEX|Nifty Midcap 50"),
            ("Nifty 100",          "NSE_INDEX|Nifty 100"),
            ("Nifty 200",          "NSE_INDEX|Nifty 200"),
            ("None — no benchmark", ""),
        ],
        "mock": [
            ("Nifty 50  (^NSEI)",  "^NSEI"),
            ("None — no benchmark", ""),
        ],
    }
    _bench_list = _BENCHMARKS.get(broker, _BENCHMARKS["yahoo"])
    _bench_labels = [label for label, _ in _bench_list]
    _bench_values = [val for _, val in _bench_list]

    _bench_sel = st.selectbox(
        "Benchmark index",
        options=_bench_labels,
        index=0,
        help=(
            "Stocks are ranked by predicted excess return vs this index. "
            "Only verified eligible tickers shown for the selected broker."
        ),
    )
    benchmark_symbol = _bench_values[_bench_labels.index(_bench_sel)]
    if benchmark_symbol:
        st.caption(f"Symbol: `{benchmark_symbol}`")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latest_trading_day() -> date:
    """Return today if weekday, else last Friday."""
    d = date.today()
    while d.weekday() >= 5:  # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d


def _resolve_symbols_file() -> Path | None:
    if use_text_symbols and symbols_text.strip():
        p = Path(data_root) / "symbols.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(symbols_text.strip() + "\n")
        return p
    sp = Path(symbols_file_path.strip()) if symbols_file_path.strip() else None
    if sp and not sp.is_absolute():
        sp = Path(__file__).resolve().parent / sp
    if sp and sp.exists():
        return sp
    fallback = Path(__file__).resolve().parent / "symbols.txt"
    return fallback if fallback.exists() else None


def _creds_flag(required_brokers=("zerodha", "upstox", "angelone")) -> str:
    if broker in required_brokers and creds_file.strip() and Path(creds_file.strip()).exists():
        return f" --creds-file {shlex.quote(creds_file.strip())}"
    return ""


def _run(cmd: str, label: str = "") -> bool:
    """Run a shell command as a list of args (no shell=True). Returns True on success."""
    import shlex as _shlex
    if label:
        st.caption(f"Running: {label}")
    with st.expander("Command", expanded=False):
        st.code(cmd, language="bash")
    with st.spinner("Working…"):
        # Split into args list to avoid shell=True injection risk
        p = subprocess.run(_shlex.split(cmd), shell=False, text=True, capture_output=True)
    if p.returncode != 0:
        st.error(p.stderr or p.stdout or "Command failed with no output.")
        return False
    if p.stdout.strip():
        with st.expander("Output", expanded=False):
            st.text(p.stdout)
    return True


def _find_latest_features() -> Path | None:
    """Return the most recently modified features parquet in data_root."""
    feat_dir = Path(data_root) / "features"
    if not feat_dir.exists():
        return None
    parquets = sorted(feat_dir.glob("features_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return parquets[0] if parquets else None


def _find_latest_ohlcv() -> Path | None:
    ohlcv_dir = Path(data_root) / "ohlcv"
    if not ohlcv_dir.exists():
        return None
    parquets = sorted(ohlcv_dir.glob("ohlcv_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return parquets[0] if parquets else None


def _find_model(kind: str, horizon: str) -> Path | None:
    p = Path("artifacts") / "models" / f"{kind}_{horizon}.joblib"
    return p if p.exists() else None


def _latest_rank_date(features_path: Path) -> date | None:
    """Return the latest date available in the features parquet."""
    try:
        from stock_knight.storage import read_df_parquet
        feats = read_df_parquet(features_path)
        dates = pd.DatetimeIndex(feats.index.get_level_values(0)).normalize().unique()
        return dates.max().date() if len(dates) else None
    except Exception:
        return None


def _rank_file_path(horizon: str, asof: date) -> Path:
    return Path("reports") / f"ranks_{horizon}_{asof}.json"


def _load_rank(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _risk_level_from_lowvol_z(lv_z: float | None) -> str:
    """Map low_volatility_z component to a coarse risk bucket."""
    if lv_z is None:
        return "—"
    if lv_z > 0.5:
        return "Low"
    if lv_z < -0.5:
        return "High"
    return "Medium"


def _build_rank_df(payload: dict, top_k: int, show_bottom: bool = False) -> pd.DataFrame:
    """Build a clean display DataFrame from rank payload."""
    ranked: list[str] = payload.get("ranked") or []
    rows_raw: list[dict] = payload.get("rows") or []
    row_map = {r.get("symbol"): r for r in rows_raw if r.get("symbol")}

    symbols = list(reversed(ranked[-top_k:])) if show_bottom else ranked[:top_k]

    records = []
    for i, sym in enumerate(symbols, 1):
        r = row_map.get(sym, {})
        pr = r.get("predicted_return")
        ltp = r.get("live_ltp")
        asof_close = r.get("asof_close")
        target = r.get("live_target_price") or r.get("asof_target_price")
        records.append(
            {
                "Rank": i,
                "Symbol": sym,
                "Live LTP ₹": round(ltp, 2) if ltp else None,
                "Close (asof) ₹": round(asof_close, 2) if asof_close else None,
                "Predicted Return %": round(pr * 100, 2) if pr is not None else None,
                "Target Price ₹": round(target, 2) if target else None,
                "Reco": r.get("recommendation") or "—",
                "Confidence %": r.get("confidence_pct"),
                "Risk": _risk_level_from_lowvol_z(r.get("low_volatility_z")),
            }
        )
    return pd.DataFrame(records)


def _apply_rank_filters(
    df: pd.DataFrame,
    *,
    reco_filter: str,
    risk_filter: str,
    pr_min: float,
    pr_max: float,
) -> pd.DataFrame:
    """Apply the Rankings-tab filters to a display DataFrame."""
    out = df
    if reco_filter and reco_filter != "All":
        out = out[out["Reco"] == reco_filter]
    if risk_filter and risk_filter != "All":
        out = out[out["Risk"] == risk_filter]
    if "Predicted Return %" in out.columns:
        out = out[
            (out["Predicted Return %"].fillna(0) >= pr_min)
            & (out["Predicted Return %"].fillna(0) <= pr_max)
        ]
    return out.reset_index(drop=True)


def _compute_raw_indicators(ohlcv_for_sym: pd.DataFrame) -> dict:
    """Compute current raw indicator values for one symbol from its OHLCV."""
    import numpy as _np

    df = ohlcv_for_sym.sort_values("date").reset_index(drop=True)
    if len(df) < 30:
        return {}
    close = df["close"].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, _np.nan)
    rsi14 = (100 - 100 / (1 + rs)).iloc[-1]

    ema12 = close.ewm(alpha=2 / 13, adjust=False).mean()
    ema26 = close.ewm(alpha=2 / 27, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist_norm = ((macd_line - macd_sig) / close.replace(0, _np.nan)).iloc[-1]

    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std(ddof=0)
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    width = (upper - lower).replace(0, _np.nan)
    bb_pct = ((close - lower) / width).iloc[-1]

    def _ma_dist(n: int) -> float | None:
        if len(close) < n:
            return None
        ma = close.rolling(n).mean().iloc[-1]
        if ma is None or pd.isna(ma) or ma == 0:
            return None
        return float(close.iloc[-1] / ma - 1.0)

    ret_1d = close.pct_change()
    vol21 = ret_1d.rolling(21).std(ddof=0).iloc[-1]

    high_252 = df["high"].astype(float).rolling(252, min_periods=63).max().iloc[-1]
    low_252 = df["low"].astype(float).rolling(252, min_periods=63).min().iloc[-1]

    return {
        "RSI_14": float(rsi14) if not pd.isna(rsi14) else None,
        "MACD_hist_norm": float(macd_hist_norm) if not pd.isna(macd_hist_norm) else None,
        "BB_pct_20": float(bb_pct) if not pd.isna(bb_pct) else None,
        "ma_dist_20": _ma_dist(20),
        "ma_dist_50": _ma_dist(50),
        "ma_dist_200": _ma_dist(200),
        "vol_21d": float(vol21) if not pd.isna(vol21) else None,
        "high_52w": float(high_252) if not pd.isna(high_252) else None,
        "low_52w": float(low_252) if not pd.isna(low_252) else None,
    }


def _generate_why(row: dict, raw: dict) -> list[str]:
    """Heuristic 3-4 bullet explanation for the Stock Detail tab."""
    bullets: list[str] = []

    mom_z = row.get("momentum_z")
    if mom_z is not None:
        if mom_z > 1.0:
            bullets.append(f"Strong 63-day momentum (z={mom_z:+.2f}, top of cross-section).")
        elif mom_z < -1.0:
            bullets.append(f"Negative 63-day momentum (z={mom_z:+.2f}, lagging peers).")

    rsi = raw.get("RSI_14")
    if rsi is not None:
        if rsi > 70:
            bullets.append(f"RSI(14)={rsi:.0f} — overbought zone.")
        elif rsi < 30:
            bullets.append(f"RSI(14)={rsi:.0f} — oversold; possible bounce.")
        else:
            bullets.append(f"RSI(14)={rsi:.0f} — neutral momentum.")

    ma50d = raw.get("ma_dist_50")
    if ma50d is not None:
        if ma50d > 0.02:
            bullets.append(f"Price {ma50d * 100:+.1f}% above 50DMA — uptrend.")
        elif ma50d < -0.02:
            bullets.append(f"Price {ma50d * 100:+.1f}% below 50DMA — downtrend.")

    lv_z = row.get("low_volatility_z")
    if lv_z is not None and lv_z > 1.0:
        bullets.append("Lower realized volatility than peers (defensive name).")
    elif lv_z is not None and lv_z < -1.0:
        bullets.append("Higher realized volatility than peers (riskier name).")

    pr = row.get("predicted_return")
    if pr is not None:
        bullets.append(f"Model predicts {pr * 100:+.2f}% forward return.")

    return bullets[:5]


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_rank, tab_detail, tab_pipeline, tab_trading = st.tabs(
    ["📊 Rankings", "🔎 Stock Detail", "⚙️ Pipeline", "💹 Trading"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — RANKINGS
# ═════════════════════════════════════════════════════════════════════════════
with tab_rank:
    st.header("Stock Rankings")
    st.caption("Stocks ranked by predicted return. Live LTP fetched from selected broker.")

    # ── Controls row ──────────────────────────────────────────────────────
    rc1, rc2, rc3, rc4 = st.columns([2, 2, 2, 4])
    with rc1:
        rank_horizon = st.selectbox("Horizon", ["1d", "7d", "1m", "6m", "12m"], index=1, key="rank_horizon")
    with rc2:
        top_k = st.selectbox("Show Top K", [10, 20, 30, 100], index=1, key="rank_topk")
    with rc3:
        show_bottom = st.checkbox("Show Bottom K too", value=False)
    with rc4:
        fetch_ltp = st.checkbox(
            "Fetch live LTP",
            value=broker in {"yahoo", "angelone", "upstox", "zerodha"},
            help="Fetches the latest traded price from the selected broker.",
        )

    # ── Rank as-of: auto-detect from features, or let user override ───────
    feat_path = _find_latest_features()
    auto_asof = None
    if feat_path:
        auto_asof = _latest_rank_date(feat_path)

    asof_default = auto_asof or _latest_trading_day()
    rank_asof = st.date_input(
        "Rank as-of date",
        value=asof_default,
        help="Auto-detected from your features data. Change if needed.",
        key="rank_asof",
    )

    # ── Action buttons ────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns([2, 3, 3])
    with btn1:
        refresh_btn = st.button("🔄 Re-Rank Now", type="primary", use_container_width=True,
                                help="Re-rank using existing features data (instant).")
    with btn2:
        quick_btn = st.button(
            "⚡ Quick Update (last 6 months + re-rank)",
            use_container_width=True,
            help="Re-ingests last 180 days of prices, rebuilds features, and re-ranks.",
        )
    with btn3:
        ltp_only_btn = st.button(
            "📡 Fetch Live Prices Only",
            use_container_width=True,
            help="Fetch today's live LTP for ranked stocks from the selected broker without re-ranking.",
        )

    # ── Fetch Live Prices only (no re-rank) ───────────────────────────
    if ltp_only_btn:
        stored_path = st.session_state.get("last_rank_path")
        ltp_rank_path = Path(stored_path) if stored_path else None
        if ltp_rank_path is None or not ltp_rank_path.exists():
            reports = sorted(Path("reports").glob(f"ranks_{rank_horizon}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if Path("reports").exists() else []
            ltp_rank_path = reports[0] if reports else None
        if ltp_rank_path is None:
            st.error("No rank file found. Run Re-Rank Now first.")
        else:
            with st.spinner("Fetching live prices from broker…"):
                try:
                    payload_raw = json.loads(ltp_rank_path.read_text())
                    ranked_syms = payload_raw.get("ranked", [])[:max(top_k, 30)]

                    # Build broker client directly in Python (no subprocess)
                    from stock_knight.credentials import load_credentials as _lc
                    _creds_data = _lc(creds_file) if Path(creds_file).exists() else {}

                    if broker == "yahoo":
                        from stock_knight.brokers.yahoo import YahooFinanceEODClient
                        _bc = YahooFinanceEODClient()
                    elif broker == "angelone":
                        from stock_knight.brokers.angelone import AngelOneConfig, AngelOneSmartApiClient
                        _a = _creds_data.get("angelone", {})
                        _missing = [k for k in ("api_key", "client_code", "password") if not _a.get(k)]
                        if _missing:
                            st.error(f"Missing Angel One credentials: {_missing}. Update credentials.yaml.")
                            _bc = None
                        else:
                            _bc = AngelOneSmartApiClient(AngelOneConfig(
                                api_key=_a["api_key"], client_code=_a["client_code"],
                                password=_a["password"], totp_secret=_a.get("totp_secret", ""),
                                totp_value=_a.get("totp_value", ""),
                            ))
                    elif broker == "upstox":
                        from stock_knight.brokers.upstox import UpstoxConfig, UpstoxEODClient
                        _u = _creds_data.get("upstox", {})
                        _at = _u.get("access_token", "")
                        if not _at or _at.startswith("YOUR_"):
                            st.error("Missing upstox.access_token in credentials.yaml.")
                            _bc = None
                        else:
                            _bc = UpstoxEODClient(UpstoxConfig(access_token=_at))
                    elif broker == "zerodha":
                        from stock_knight.brokers.zerodha import ZerodhaConfig, ZerodhaKiteEODClient
                        _z = _creds_data.get("zerodha", {})
                        _missing = [k for k in ("api_key", "access_token") if not _z.get(k)]
                        if _missing:
                            st.error(f"Missing Zerodha credentials: {_missing}. Update credentials.yaml.")
                            _bc = None
                        else:
                            _bc = ZerodhaKiteEODClient(ZerodhaConfig(
                                api_key=_z["api_key"], access_token=_z["access_token"]
                            ))
                    else:
                        st.warning("Live prices not available for mock broker.")
                        _bc = None

                    if _bc:
                        ltp_map = _bc.fetch_ltp(ranked_syms)
                        # Patch the payload rows with fresh LTP
                        from datetime import UTC, datetime as _dt
                        now_utc = _dt.now(UTC).replace(microsecond=0).isoformat()
                        for row in payload_raw.get("rows", []):
                            sym = row.get("symbol", "")
                            if sym in ltp_map:
                                pr = row.get("predicted_return", 0) or 0
                                row["live_ltp"] = ltp_map[sym]
                                row["live_target_price"] = ltp_map[sym] * (1 + pr)
                        payload_raw["live_ltp_fetched_at"] = now_utc
                        payload_raw["live_ltp_broker"] = broker
                        ltp_rank_path.write_text(json.dumps(payload_raw, indent=2))
                        st.session_state["last_rank_path"] = str(ltp_rank_path)
                        fetched = len(ltp_map)
                        st.success(f"Live prices fetched for {fetched} symbols at {now_utc[:16]} UTC")
                except Exception as e:
                    st.error(f"Live price fetch failed: {e}")

    if refresh_btn or quick_btn:
        feat_path = _find_latest_features()
        ohlcv_path = _find_latest_ohlcv()
        model_path = _find_model(model_kind, rank_horizon)
        syms_file = _resolve_symbols_file()

        # ── Quick Update: re-ingest last 6 months ─────────────────────
        if quick_btn:
            quick_start = (date.today() - timedelta(days=180)).isoformat()
            quick_end = date.today().isoformat()
            if broker in {"yahoo"} and syms_file is None:
                st.error("Symbols file required for Yahoo broker.")
                st.stop()
            extra = f" --symbols-file {shlex.quote(str(syms_file))}" if syms_file else ""
            ok = _run(
                f"python -m stock_knight.cli ingest --broker {shlex.quote(broker)} "
                f"--data-root {shlex.quote(data_root)} "
                f"--start {quick_start} --end {quick_end}"
                f"{_creds_flag()}{extra}"
                + (f" --benchmark-symbol {shlex.quote(benchmark_symbol)}" if train_target == "excess" and benchmark_symbol.strip() else ""),
                label="Re-ingesting last 6 months",
            )
            if not ok:
                st.stop()
            # Rebuild features from the full existing OHLCV
            ohlcv_path = _find_latest_ohlcv()
            if ohlcv_path:
                ok = _run(
                    f"python -m stock_knight.cli features-labels "
                    f"--data-root {shlex.quote(data_root)} "
                    f"--ohlcv-parquet {shlex.quote(str(ohlcv_path))}"
                    + (f" --benchmark-symbol {shlex.quote(benchmark_symbol)}" if train_target == "excess" and benchmark_symbol.strip() else ""),
                    label="Rebuilding features",
                )
                if not ok:
                    st.stop()
                feat_path = _find_latest_features()
                ohlcv_path = _find_latest_ohlcv()
                # Re-detect latest asof
                if feat_path:
                    auto_asof = _latest_rank_date(feat_path)
                    if auto_asof:
                        rank_asof = auto_asof
            else:
                st.warning("No OHLCV data found after re-ingest.")

        # ── Rank ──────────────────────────────────────────────────────
        if feat_path is None or not feat_path.exists():
            st.error("No features parquet found. Run the Pipeline tab first.")
            st.stop()
        if model_path is None:
            st.error(f"Model not found: artifacts/models/{model_kind}_{rank_horizon}.joblib. Run Pipeline → Train.")
            st.stop()

        ohlcv_path = _find_latest_ohlcv()
        out_path = _rank_file_path(rank_horizon, rank_asof)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ltp_part = ""
        if fetch_ltp and broker in {"yahoo", "angelone", "upstox", "zerodha"}:
            ltp_broker = "yahoo" if broker == "yahoo" else broker
            ltp_part = f" --live-ltp --ltp-broker {shlex.quote(ltp_broker)}"
            if broker in {"zerodha", "angelone", "upstox"}:
                ltp_part += _creds_flag()

        ok = _run(
            "python -m stock_knight.cli rank "
            f"--features-parquet {shlex.quote(str(feat_path))} "
            f"--ohlcv-parquet {shlex.quote(str(ohlcv_path))} "
            f"--model-path {shlex.quote(str(model_path))} "
            f"--asof {rank_asof} --horizon {rank_horizon} "
            f"--training-target {train_target}"
            + (f" --benchmark-symbol {shlex.quote(benchmark_symbol)}" if train_target == "excess" and benchmark_symbol.strip() else "")
            + ltp_part
            + f" --out {shlex.quote(str(out_path))}",
            label="Ranking stocks",
        )
        if ok:
            st.session_state["last_rank_path"] = str(out_path)
            st.success(f"Rankings updated for {rank_asof}")

    # ── Load & display rank results ──────────────────────────────────────
    stored_path = st.session_state.get("last_rank_path")
    rank_path = Path(stored_path) if stored_path else _rank_file_path(rank_horizon, rank_asof)

    # Try to find any existing rank file if the exact one doesn't exist
    if not rank_path.exists():
        reports = sorted(Path("reports").glob(f"ranks_{rank_horizon}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True) if Path("reports").exists() else []
        if reports:
            rank_path = reports[0]

    payload = _load_rank(rank_path)
    if payload:
        # Metadata bar
        meta1, meta2, meta3, meta4 = st.columns(4)
        meta1.metric("Horizon", payload.get("horizon", "—").upper())
        effective_date = str(payload.get("asof", "—"))[:10]
        meta2.metric("Features as-of", effective_date)
        meta3.metric("Training Target", payload.get("training_target", "—"))
        ltp_ts = payload.get("live_ltp_fetched_at")
        meta4.metric("LTP Fetched", ltp_ts[:16] + " UTC" if ltp_ts else "Historical close only")

        # Warn when features are stale (older than 5 trading days)
        try:
            feat_date = pd.to_datetime(effective_date).date()
            days_old = (date.today() - feat_date).days
            if days_old > 5:
                st.warning(
                    f"⚠️ Model features are **{days_old} days old** (last: {effective_date}). "
                    "Use **⚡ Quick Update** to get fresh features. "
                    "For today's prices, click **📡 Fetch Live Prices Only**."
                )
        except Exception:
            pass

        # Show whether live prices are available
        rows_raw = payload.get("rows", [])
        has_live = any(r.get("live_ltp") for r in rows_raw)
        if not has_live:
            st.info("💡 Prices shown are historical (from model's last data date). Click **📡 Fetch Live Prices Only** to load today's market prices.")

        st.divider()

        # ── Filters ───────────────────────────────────────────────────────
        st.markdown("**Filters**")
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 3, 1.4])
        with f1:
            reco_filter = st.selectbox(
                "Recommendation",
                ["All", "BUY", "HOLD", "SELL"],
                index=0,
                key="rank_reco_filter",
            )
        with f2:
            risk_filter = st.selectbox(
                "Risk level",
                ["All", "Low", "Medium", "High"],
                index=0,
                help="Derived from low-volatility z-score component of the composite score.",
                key="rank_risk_filter",
            )
        with f3:
            pr_min, pr_max = st.slider(
                "Predicted return range %",
                min_value=-50.0,
                max_value=50.0,
                value=(-50.0, 50.0),
                step=0.5,
                key="rank_pr_range",
            )
        with f4:
            custom_topk = st.number_input(
                "Custom Top-K",
                min_value=1,
                max_value=500,
                value=int(top_k),
                step=1,
                help="Overrides the Top-K dropdown above.",
                key="rank_custom_topk",
            )
        effective_topk = int(custom_topk)

        def _color_return(val):
            if val is None:
                return ""
            return "color: green; font-weight: bold" if val > 0 else "color: red; font-weight: bold"

        def _color_reco(val):
            if val == "BUY":
                return "color: green; font-weight: bold"
            if val == "SELL":
                return "color: red; font-weight: bold"
            return ""

        # Top K table
        top_df = _build_rank_df(payload, effective_topk, show_bottom=False)
        top_df = _apply_rank_filters(
            top_df,
            reco_filter=reco_filter,
            risk_filter=risk_filter,
            pr_min=pr_min,
            pr_max=pr_max,
        )
        if not top_df.empty:
            price_label = "📡 Live" if has_live else f"📅 Historical ({effective_date})"
            st.markdown(
                f"### 🔼 Top {effective_topk} — Best predicted return "
                f"({rank_horizon.upper()}) | Price: {price_label}"
            )

            fmt_cols = {
                "Close (asof) ₹": "{:.2f}",
                "Predicted Return %": "{:+.2f}%",
                "Target Price ₹": "{:.2f}",
                "Confidence %": "{:.1f}%",
            }
            if has_live:
                fmt_cols["Live LTP ₹"] = "{:.2f}"

            cols_to_show = list(top_df.columns)
            if has_live and "Close (asof) ₹" in cols_to_show:
                cols_to_show = [c for c in cols_to_show if c != "Close (asof) ₹"]

            styled = (
                top_df[cols_to_show]
                .style.map(_color_return, subset=["Predicted Return %"])
                .map(_color_reco, subset=["Reco"])
                .format(
                    {k: v for k, v in fmt_cols.items() if k in cols_to_show},
                    na_rep="—",
                )
            )
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=min(42 * effective_topk + 60, 700),
            )
        else:
            st.info("No rows match the current filters.")

        # Bottom K table
        if show_bottom:
            bot_df = _build_rank_df(payload, effective_topk, show_bottom=True)
            bot_df = _apply_rank_filters(
                bot_df,
                reco_filter=reco_filter,
                risk_filter=risk_filter,
                pr_min=pr_min,
                pr_max=pr_max,
            )
            if not bot_df.empty:
                st.markdown(
                    f"### 🔽 Bottom {effective_topk} — Worst predicted return "
                    f"({rank_horizon.upper()})"
                )
                cols_to_show = list(bot_df.columns)
                if has_live and "Close (asof) ₹" in cols_to_show:
                    cols_to_show = [c for c in cols_to_show if c != "Close (asof) ₹"]
                styled_bot = (
                    bot_df[cols_to_show]
                    .style.map(_color_return, subset=["Predicted Return %"])
                    .map(_color_reco, subset=["Reco"])
                    .format(
                        {k: v for k, v in fmt_cols.items() if k in cols_to_show},
                        na_rep="—",
                    )
                )
                st.dataframe(styled_bot, use_container_width=True, hide_index=True)

        st.caption(f"Source: `{rank_path}`")
    else:
        st.info(
            "No rank file found yet. Click **🔄 Re-Rank Now** above, "
            "or run the full pipeline in the **⚙️ Pipeline** tab first."
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — STOCK DETAIL
# ═════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.header("Stock Detail")
    st.caption(
        "Per-symbol drill-down: candlestick chart, indicators, recommendation, "
        "and a generated explanation of the model's view."
    )

    detail_payload: dict | None = None
    detail_path: Path | None = None
    stored_path = st.session_state.get("last_rank_path")
    if stored_path:
        p = Path(stored_path)
        if p.exists():
            detail_payload = _load_rank(p)
            detail_path = p
    if detail_payload is None:
        # fall back to latest report
        if Path("reports").exists():
            reports = sorted(
                Path("reports").glob("ranks_*_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if reports:
                detail_path = reports[0]
                detail_payload = _load_rank(detail_path)

    if not detail_payload:
        st.info(
            "No rank file available yet. Run **🔄 Re-Rank Now** in the Rankings tab "
            "to populate this view."
        )
    else:
        ranked_syms = detail_payload.get("ranked") or []
        rows_raw = detail_payload.get("rows") or []
        row_map = {r.get("symbol"): r for r in rows_raw if r.get("symbol")}
        horizon_label = (detail_payload.get("horizon") or "").upper() or "—"

        if not ranked_syms:
            st.warning("Rank file has no ranked symbols.")
        else:
            d_top = st.columns([3, 1, 1])
            with d_top[0]:
                sel_symbol = st.selectbox(
                    "Symbol",
                    ranked_syms,
                    index=0,
                    key="detail_symbol",
                )
            with d_top[1]:
                st.metric("Horizon", horizon_label)
            with d_top[2]:
                rank_idx = ranked_syms.index(sel_symbol) + 1
                st.metric("Rank", f"{rank_idx} / {len(ranked_syms)}")

            row = row_map.get(sel_symbol, {})

            # ── Headline cards ────────────────────────────────────────────
            card_cols = st.columns(5)
            ltp = row.get("live_ltp") or row.get("asof_close")
            target = row.get("live_target_price") or row.get("asof_target_price")
            pr = row.get("predicted_return")
            reco = row.get("recommendation") or "—"
            conf = row.get("confidence_pct")

            card_cols[0].metric("Price ₹", f"{ltp:,.2f}" if ltp else "—")
            card_cols[1].metric(
                "Target ₹", f"{target:,.2f}" if target else "—",
                delta=(f"{pr * 100:+.2f}%" if pr is not None else None),
            )
            card_cols[2].metric(
                "Recommendation", reco,
                help="BUY = top quintile, SELL = bottom quintile, else HOLD.",
            )
            card_cols[3].metric(
                "Confidence",
                f"{conf:.1f}%" if conf is not None else "—",
                help="|sort_score sigma| / 3, capped at 100%.",
            )
            card_cols[4].metric(
                "Risk",
                _risk_level_from_lowvol_z(row.get("low_volatility_z")),
                help="From low-volatility z-score component.",
            )

            # ── Chart ─────────────────────────────────────────────────────
            ohlcv_path = _find_latest_ohlcv()
            if ohlcv_path is None:
                st.warning("No OHLCV parquet found — chart unavailable.")
                sym_ohlcv = pd.DataFrame()
            else:
                try:
                    from stock_knight.storage import read_df_parquet as _rdp

                    full_ohlcv = _rdp(ohlcv_path)
                    sym_ohlcv = full_ohlcv[full_ohlcv["symbol"] == sel_symbol].copy()
                    sym_ohlcv["date"] = pd.to_datetime(sym_ohlcv["date"]).dt.tz_localize(None)
                    sym_ohlcv = sym_ohlcv.sort_values("date").tail(252)
                except Exception as e:
                    st.error(f"Failed to read OHLCV: {e}")
                    sym_ohlcv = pd.DataFrame()

            raw_inds = _compute_raw_indicators(sym_ohlcv) if not sym_ohlcv.empty else {}

            try:
                import plotly.graph_objects as go

                if not sym_ohlcv.empty:
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=sym_ohlcv["date"],
                                open=sym_ohlcv["open"],
                                high=sym_ohlcv["high"],
                                low=sym_ohlcv["low"],
                                close=sym_ohlcv["close"],
                                name=sel_symbol,
                            )
                        ]
                    )
                    if target:
                        fig.add_hline(
                            y=target,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text=f"Target ₹{target:,.2f}",
                            annotation_position="top right",
                        )
                    if raw_inds.get("high_52w"):
                        fig.add_hline(
                            y=raw_inds["high_52w"],
                            line_dash="dot",
                            line_color="green",
                            annotation_text="52w high",
                            annotation_position="top left",
                        )
                    if raw_inds.get("low_52w"):
                        fig.add_hline(
                            y=raw_inds["low_52w"],
                            line_dash="dot",
                            line_color="red",
                            annotation_text="52w low",
                            annotation_position="bottom left",
                        )
                    fig.update_layout(
                        height=460,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis_rangeslider_visible=False,
                        title=f"{sel_symbol} — last {len(sym_ohlcv)} sessions",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning(
                    "`plotly` is not installed. Install with "
                    "`pip install -e '.[ui]'` to see the candlestick chart."
                )
                if not sym_ohlcv.empty:
                    st.line_chart(
                        sym_ohlcv.set_index("date")[["close"]],
                        height=400,
                    )

            # ── Indicators panel ──────────────────────────────────────────
            st.markdown("### Indicators")
            ind_cols = st.columns(7)

            def _fmt_pct(v: float | None) -> str:
                return f"{v * 100:+.2f}%" if v is not None else "—"

            def _fmt_num(v: float | None, digits: int = 2) -> str:
                return f"{v:.{digits}f}" if v is not None else "—"

            ind_cols[0].metric("RSI(14)", _fmt_num(raw_inds.get("RSI_14"), 1))
            ind_cols[1].metric(
                "MACD hist (norm)", _fmt_pct(raw_inds.get("MACD_hist_norm"))
            )
            ind_cols[2].metric("BB %", _fmt_num(raw_inds.get("BB_pct_20"), 2))
            ind_cols[3].metric("vs 20DMA", _fmt_pct(raw_inds.get("ma_dist_20")))
            ind_cols[4].metric("vs 50DMA", _fmt_pct(raw_inds.get("ma_dist_50")))
            ind_cols[5].metric("vs 200DMA", _fmt_pct(raw_inds.get("ma_dist_200")))
            ind_cols[6].metric("Vol(21d)", _fmt_pct(raw_inds.get("vol_21d")))

            # ── Composite breakdown ───────────────────────────────────────
            if "composite" in row:
                st.markdown("### Composite score breakdown")
                cb_cols = st.columns(6)
                cb_cols[0].metric("Composite", _fmt_num(row.get("composite"), 3))
                cb_cols[1].metric("Return z", _fmt_num(row.get("expected_return_z"), 2))
                cb_cols[2].metric("Sharpe z", _fmt_num(row.get("sharpe_z"), 2))
                cb_cols[3].metric("Momentum z", _fmt_num(row.get("momentum_z"), 2))
                cb_cols[4].metric("Low-vol z", _fmt_num(row.get("low_volatility_z"), 2))
                cb_cols[5].metric(
                    "Confidence z", _fmt_num(row.get("model_confidence_z"), 2)
                )
                weights = detail_payload.get("composite_weights") or {}
                if weights:
                    st.caption(
                        "Weights — "
                        + " · ".join(f"{k}: {v}" for k, v in weights.items())
                    )

            # ── Why explanation ───────────────────────────────────────────
            st.markdown("### Why this view")
            bullets = _generate_why(row, raw_inds)
            if bullets:
                for b in bullets:
                    st.markdown(f"- {b}")
            else:
                st.caption("No interpretable signals — try reranking with a longer feature history.")

            if detail_path is not None:
                st.caption(f"Source: `{detail_path}`")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.header("Data & Training Pipeline")
    st.caption(
        "Run these steps once to set up the model, then use **Quick Update** in Rankings "
        "for daily refreshes."
    )

    # Status summary
    feat_path = _find_latest_features()
    ohlcv_path = _find_latest_ohlcv()
    models_exist = {
        h: (Path("artifacts") / "models" / f"{model_kind}_{h}.joblib").exists()
        for h in ["1d", "7d", "1m", "6m", "12m"]
    }

    s1, s2, s3 = st.columns(3)
    s1.metric("OHLCV data", "✅ Found" if ohlcv_path else "❌ Missing", delta=str(ohlcv_path.name) if ohlcv_path else "run Ingest")
    s2.metric("Features", "✅ Found" if feat_path else "❌ Missing", delta=str(feat_path.name) if feat_path else "run Features/Labels")
    trained = [h for h, v in models_exist.items() if v]
    s3.metric("Models", f"✅ {len(trained)}/5" if trained else "❌ None", delta=", ".join(trained) if trained else "run Train")

    st.divider()

    syms_file = _resolve_symbols_file()

    # ── Step 1: Ingest ────────────────────────────────────────────────────
    with st.expander("Step 1 — Ingest EOD data", expanded=not bool(ohlcv_path)):
        st.write(f"Fetches daily OHLCV from **{broker}** for all symbols in `{syms_file or 'symbols.txt'}`.")
        p_start = st.date_input("From", value=start, key="pipe_start")
        p_end = st.date_input("To", value=end, key="pipe_end")
        if st.button("Run Ingest", key="btn_ingest", type="primary"):
            if broker in {"yahoo"} and syms_file is None:
                st.error("Symbols file required for Yahoo broker.")
                st.stop()
            extra = f" --symbols-file {shlex.quote(str(syms_file))}" if syms_file else ""
            ok = _run(
                f"python -m stock_knight.cli ingest --broker {shlex.quote(broker)} "
                f"--data-root {shlex.quote(data_root)} "
                f"--start {p_start} --end {p_end}"
                f"{_creds_flag()}{extra}"
                + (f" --benchmark-symbol {shlex.quote(benchmark_symbol)}" if train_target == "excess" and benchmark_symbol.strip() else ""),
                label="Ingest",
            )
            if ok:
                st.success("Ingest complete.")

    # ── Step 2: Features / Labels ─────────────────────────────────────────
    with st.expander("Step 2 — Build features & labels", expanded=bool(ohlcv_path) and not bool(feat_path)):
        ohlcv_path = _find_latest_ohlcv()
        if ohlcv_path:
            st.write(f"Using OHLCV: `{ohlcv_path.name}`")
        else:
            st.warning("No OHLCV parquet found. Run Step 1 first.")
        if st.button("Run Features + Labels", key="btn_fl", type="primary"):
            if not ohlcv_path:
                st.error("No OHLCV data found.")
                st.stop()
            ok = _run(
                f"python -m stock_knight.cli features-labels "
                f"--data-root {shlex.quote(data_root)} "
                f"--ohlcv-parquet {shlex.quote(str(ohlcv_path))}"
                + (f" --benchmark-symbol {shlex.quote(benchmark_symbol)}" if train_target == "excess" and benchmark_symbol.strip() else ""),
                label="Features & Labels",
            )
            if ok:
                st.success("Features and labels built.")

    # ── Step 3: Train ─────────────────────────────────────────────────────
    with st.expander("Step 3 — Train model", expanded=bool(feat_path) and not any(models_exist.values())):
        feat_path = _find_latest_features()
        if feat_path:
            st.write(f"Using features: `{feat_path.name}`")
        else:
            st.warning("No features parquet found. Run Step 2 first.")
        if st.button("Run Train (all horizons)", key="btn_train", type="primary"):
            if not feat_path:
                st.error("No features data found.")
                st.stop()
            lab_path = Path(str(feat_path).replace("features_", "labels_").replace("/features/", "/labels/"))
            if not lab_path.exists():
                st.error(f"Labels not found: `{lab_path}`. Re-run Step 2.")
                st.stop()
            ok = _run(
                f"python -m stock_knight.cli train "
                f"--data-root {shlex.quote(data_root)} "
                f"--features-parquet {shlex.quote(str(feat_path))} "
                f"--labels-parquet {shlex.quote(str(lab_path))} "
                f"--model-kind {model_kind} --target {train_target}",
                label="Training models",
            )
            if ok:
                st.success("Training complete. Models saved in `artifacts/models/`")

    # ── Step 4: Rank ──────────────────────────────────────────────────────
    with st.expander("Step 4 — Generate rank file", expanded=False):
        feat_path = _find_latest_features()
        ohlcv_path = _find_latest_ohlcv()
        pipe_horizon = st.selectbox("Horizon", ["1d", "7d", "1m", "6m", "12m"], index=1, key="pipe_horizon")
        pipe_asof = st.date_input(
            "As-of date",
            value=_latest_trading_day(),
            key="pipe_asof",
        )
        if st.button("Run Rank", key="btn_rank", type="primary"):
            if not feat_path:
                st.error("No features data found.")
                st.stop()
            model_path = _find_model(model_kind, pipe_horizon)
            if not model_path:
                st.error(f"Model not found for horizon={pipe_horizon}. Run Step 3.")
                st.stop()
            out_path = _rank_file_path(pipe_horizon, pipe_asof)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ltp_part = ""
            if broker in {"yahoo", "zerodha", "angelone", "upstox"}:
                ltp_broker = "yahoo" if broker == "yahoo" else broker
                ltp_part = f" --live-ltp --ltp-broker {shlex.quote(ltp_broker)}"
                if broker in {"zerodha", "angelone", "upstox"}:
                    ltp_part += _creds_flag()
            ok = _run(
                "python -m stock_knight.cli rank "
                f"--features-parquet {shlex.quote(str(feat_path))} "
                f"--ohlcv-parquet {shlex.quote(str(ohlcv_path))} "
                f"--model-path {shlex.quote(str(model_path))} "
                f"--asof {pipe_asof} --horizon {pipe_horizon} "
                f"--training-target {train_target}"
                + (f" --benchmark-symbol {shlex.quote(benchmark_symbol)}" if train_target == "excess" and benchmark_symbol.strip() else "")
                + ltp_part
                + f" --out {shlex.quote(str(out_path))}",
                label="Ranking",
            )
            if ok:
                st.session_state["last_rank_path"] = str(out_path)
                st.success(f"Rank file written: `{out_path}`. Switch to Rankings tab to view.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADING
# ═════════════════════════════════════════════════════════════════════════════
with tab_trading:
    st.header("Live Order Execution")

    TRADING_BROKERS = {"zerodha", "angelone", "upstox"}
    if broker not in TRADING_BROKERS:
        st.info(
            f"Trading is available for **Angel One**, **Zerodha**, and **Upstox**. "
            f"Current broker is **{broker}**. Switch in the sidebar."
        )
    else:
        st.caption(
            f"Broker: **{broker.title()}** — "
            "Place equity orders based on the latest rank output. "
            "Always start with a dry-run before going live."
        )

        def _get_trading_client():
            cf = creds_file.strip()
            if not cf or not Path(cf).exists():
                st.error(f"Credentials file not found: `{cf}`")
                return None
            try:
                from stock_knight.credentials import load_credentials
                creds_data = load_credentials(cf)

                if broker == "angelone":
                    from stock_knight.brokers.angelone import AngelOneConfig, AngelOneSmartApiClient
                    a = creds_data.get("angelone", {})
                    if not a.get("api_key") or a.get("api_key", "").startswith("YOUR_"):
                        st.error("Missing angelone.api_key in credentials.yaml")
                        return None
                    with st.spinner("Connecting to Angel One…"):
                        return AngelOneSmartApiClient(AngelOneConfig(
                            api_key=a["api_key"],
                            client_code=a.get("client_code", ""),
                            password=a.get("password", ""),
                            totp_secret=a.get("totp_secret", ""),
                            totp_value=a.get("totp_value", ""),
                        ))

                if broker == "zerodha":
                    from stock_knight.brokers.zerodha import ZerodhaConfig, ZerodhaKiteEODClient
                    z = creds_data.get("zerodha", {})
                    at = z.get("access_token", "")
                    if not at or at == "YOUR_ZERODHA_ACCESS_TOKEN":
                        st.error("Missing/placeholder zerodha.access_token. Run `zerodha-login` first.")
                        return None
                    return ZerodhaKiteEODClient(ZerodhaConfig(
                        api_key=z.get("api_key", ""),
                        access_token=at,
                    ))

                if broker == "upstox":
                    from stock_knight.brokers.upstox import UpstoxConfig, UpstoxEODClient
                    u = creds_data.get("upstox", {})
                    at = u.get("access_token", "")
                    if not at or at.startswith("YOUR_"):
                        st.error("Missing upstox.access_token in credentials.yaml")
                        return None
                    return UpstoxEODClient(UpstoxConfig(access_token=at))

            except Exception as e:
                st.error(f"Connection failed: {e}")
                return None

        def _get_trader(client):
            if broker == "angelone":
                from stock_knight.execution.angelone_trader import AngelOneTrader
                return AngelOneTrader(client=client, product=trade_product, order_type=trade_order_type)
            if broker == "zerodha":
                from stock_knight.execution.zerodha_trader import ZerodhaTrader
                return ZerodhaTrader(client=client, product=trade_product, order_type=trade_order_type)
            # Upstox uses its own order API — trading not yet wired to a dedicated Trader class.
            st.warning(f"Automated trade execution is not yet supported for broker '{broker}'. "
                       "Use Zerodha or Angel One for live order placement.")
            return None

        # ── Trading controls ──────────────────────────────────────────────
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            trade_top_k = st.number_input("Top-K to trade", min_value=1, max_value=50, value=10)
            trade_capital = st.number_input("Capital ₹ (0 = use all available)", min_value=0, value=0, step=5000)
            sell_exits = st.checkbox("Sell stocks no longer in top-K", value=True)
        with tc2:
            trade_order_type = st.selectbox("Order type", ["MARKET", "LIMIT"])
            product_options = {"angelone": ["DELIVERY", "INTRADAY"], "zerodha": ["CNC", "MIS"], "upstox": ["D", "I"]}
            trade_product = st.selectbox("Product", product_options.get(broker, ["CNC", "MIS"]))
        with tc3:
            rank_file_for_trade = st.text_input(
                "Rank file",
                value=st.session_state.get("last_rank_path", str(_rank_file_path(
                    st.session_state.get("rank_horizon", "7d"),
                    _latest_trading_day()
                ))),
            )

        st.divider()

        # ── Funds & Holdings refresh ──────────────────────────────────────
        if st.button("Refresh Funds & Holdings", key="trade_refresh"):
            client = _get_trading_client()
            if client:
                try:
                    st.session_state["t_funds"] = client.get_funds()
                    st.session_state["t_holdings"] = client.get_holdings()
                    st.session_state["t_positions"] = client.get_positions()
                    st.success("Loaded.")
                except Exception as e:
                    st.error(f"{e}")

        if "t_funds" in st.session_state:
            f = st.session_state["t_funds"]
            fm1, fm2, fm3 = st.columns(3)
            fm1.metric("Available Cash", f"₹{f.get('available_cash', 0):,.0f}")
            fm2.metric("Used Margin", f"₹{f.get('used_margin', 0):,.0f}")
            fm3.metric("Net", f"₹{f.get('net', 0):,.0f}")

        if "t_holdings" in st.session_state and not st.session_state["t_holdings"].empty:
            st.markdown("**Holdings**")
            st.dataframe(st.session_state["t_holdings"], use_container_width=True, hide_index=True)

        if "t_positions" in st.session_state and not st.session_state["t_positions"].empty:
            st.markdown("**Open positions**")
            st.dataframe(st.session_state["t_positions"], use_container_width=True, hide_index=True)

        st.divider()

        # ── Order plan ────────────────────────────────────────────────────
        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button("📋 Preview order plan", key="trade_preview", use_container_width=True):
                rf = Path(rank_file_for_trade)
                if not rf.exists():
                    st.error(f"Rank file not found: `{rf}`. Run Rankings → Re-Rank Now first.")
                else:
                    client = _get_trading_client()
                    if client:
                        try:
                            trader = _get_trader(client)
                            plan = trader.build_plan(
                                rank_file=rf,
                                top_k=int(trade_top_k),
                                capital=float(trade_capital) if trade_capital > 0 else None,
                                sell_exits=sell_exits,
                            )
                            st.session_state["t_plan"] = plan
                            st.success(
                                f"Plan: {len(plan.orders)} order(s) | "
                                f"Buy ₹{plan.total_buy_value:,.0f} | Sell ₹{plan.total_sell_value:,.0f}"
                            )
                        except Exception as e:
                            st.error(f"{e}")

        with pc2:
            live_exec = st.checkbox("Enable LIVE execution ⚠️", value=False,
                                    help="Uncheck = dry-run. Check = real orders on exchange.")
            if live_exec:
                st.warning("LIVE MODE — real orders will be placed on the exchange!")
            exec_label = "🚀 Execute LIVE orders" if live_exec else "🧪 Execute (dry-run)"
            if st.button(exec_label, key="trade_exec", use_container_width=True,
                         type="primary" if live_exec else "secondary"):
                if "t_plan" not in st.session_state:
                    st.warning("Preview the plan first.")
                else:
                    client = _get_trading_client()
                    if client:
                        try:
                            trader = _get_trader(client)
                            plan = trader.execute(st.session_state["t_plan"], dry_run=not live_exec)
                            st.session_state["t_executed"] = plan
                            placed = sum(1 for o in plan.orders if o.status == "PLACED")
                            failed = sum(1 for o in plan.orders if o.status == "FAILED")
                            dry = sum(1 for o in plan.orders if o.status == "DRY_RUN")
                            if live_exec:
                                st.success(f"{placed} placed / {failed} failed")
                            else:
                                st.info(f"Dry-run: {dry} order(s) simulated")
                        except Exception as e:
                            st.error(f"{e}")

        # ── Show plan table ───────────────────────────────────────────────
        plan_show = st.session_state.get("t_executed") or st.session_state.get("t_plan")
        if plan_show:
            pd_ = plan_show.to_dict()
            if pd_["orders"]:
                st.markdown(
                    f"**Order plan** | Capital ₹{pd_['capital_allocated']:,.0f} | "
                    f"Available ₹{pd_['available_cash']:,.0f} | "
                    f"Buy ₹{pd_['total_buy_value']:,.0f} | Sell ₹{pd_['total_sell_value']:,.0f}"
                )
                orders_df = pd.DataFrame(pd_["orders"])
                st.dataframe(orders_df, use_container_width=True, hide_index=True)
            if pd_["skipped"]:
                with st.expander(f"Skipped ({len(pd_['skipped'])})"):
                    st.dataframe(pd.DataFrame(pd_["skipped"]), use_container_width=True, hide_index=True)

        # ── Today's orders ────────────────────────────────────────────────
        if st.button("Refresh today's orders", key="trade_orders_refresh"):
            client = _get_trading_client()
            if client:
                try:
                    st.session_state["t_orders"] = client.get_orders()
                except Exception as e:
                    st.error(f"{e}")
        if "t_orders" in st.session_state:
            odf = st.session_state["t_orders"]
            st.markdown("**Today's orders**")
            st.dataframe(odf if not odf.empty else pd.DataFrame([{"info": "No orders today"}]),
                         use_container_width=True, hide_index=True)
