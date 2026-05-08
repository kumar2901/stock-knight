"""
Optional Yahoo Finance (yfinance) fundamentals for feature enrichment.

Fetches a **point-in-time snapshot** at build time (``Ticker.info`` + latest
``balance_sheet``). The same values are broadcast to every `(date, symbol)` row
for that symbol — fine for **live cross-sectional ranking**; **walk-forward
backtests** can be biased vs true historical fundamentals. Prefer vendor
point-in-time data for research-grade training.

All model-facing columns end with ``_z`` (cross-sectional z-score per date).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from stock_knight.logger import get_logger

_log = get_logger(__name__)

# Rate limit: Yahoo may throttle aggressive parallel requests.
_DEFAULT_DELAY_S = 0.2

_RAW_COLS = [
    "yf_trailing_pe",
    "yf_forward_pe",
    "yf_pb",
    "yf_dividend_yield",
    "yf_debt_to_equity",
    "yf_current_ratio",
    "yf_roe",
    "yf_profit_margin",
    "yf_current_ratio_bs",
    "yf_debt_to_equity_bs",
]


def _to_nse_yahoo_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s or s.startswith("^"):
        return s
    if s.endswith((".NS", ".BO")):
        return s
    return f"{s}.NS"


def _cs_zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def _pick_bs(bs: pd.DataFrame, names: tuple[str, ...], col) -> float | None:
    for n in names:
        if n in bs.index:
            v = bs.loc[n, col]
            if pd.isna(v):
                return None
            return float(v)
    return None


def _balance_sheet_metrics(ticker: Any) -> dict[str, float | None]:
    """Ratios from the most recent balance-sheet column (if available)."""
    out: dict[str, float | None] = {
        "yf_current_ratio_bs": None,
        "yf_debt_to_equity_bs": None,
    }
    try:
        bs = ticker.balance_sheet
        if bs is None or bs.empty or len(bs.columns) < 1:
            return out
        c0 = bs.columns[0]
        ca = _pick_bs(bs, ("Current Assets", "Total Current Assets"), c0)
        cl = _pick_bs(bs, ("Current Liabilities", "Total Current Liabilities"), c0)
        if ca is not None and cl is not None and cl != 0:
            out["yf_current_ratio_bs"] = ca / cl
        if out["yf_current_ratio_bs"] is None:
            # Many NSE names omit currentRatio in ``info``; Yahoo may only expose
            # Working Capital + Current Liabilities (CA/CL = 1 + WC/CL).
            wc = _pick_bs(bs, ("Working Capital",), c0)
            cl_wc = cl if cl is not None else _pick_bs(
                bs, ("Current Liabilities", "Total Current Liabilities"), c0
            )
            if wc is not None and cl_wc is not None and cl_wc != 0:
                out["yf_current_ratio_bs"] = 1.0 + (wc / cl_wc)

        debt = _pick_bs(
            bs,
            ("Total Debt", "Long Term Debt And Capital Lease Obligation", "Total Liabilities Net Minority Interest"),
            c0,
        )
        eq = _pick_bs(
            bs,
            (
                "Stockholders Equity",
                "Common Stock Equity",
                "Total Stockholder Equity",
            ),
            c0,
        )
        if debt is not None and eq is not None and eq != 0:
            out["yf_debt_to_equity_bs"] = debt / abs(eq)
    except Exception as e:  # pragma: no cover
        _log.debug("balance_sheet parse skip: %s", e)
    return out


def _roe_from_statements(ticker: Any) -> float | None:
    """ROE = net income / equity from latest annual statements if ``info`` omits ROE."""
    try:
        fin = ticker.financials
        bs = ticker.balance_sheet
        if fin is None or fin.empty or bs is None or bs.empty:
            return None
        fc = fin.columns[0]
        bc = bs.columns[0]
        ni: float | None = None
        for name in (
            "Net Income Common Stockholders",
            "Net Income",
            "Net Income From Continuing Operation Net Minority Interest",
        ):
            if name not in fin.index:
                continue
            v = fin.loc[name, fc]
            if pd.notna(v):
                ni = float(v)
                break
        eq: float | None = None
        for name in (
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Stockholder Equity",
        ):
            if name not in bs.index:
                continue
            v = bs.loc[name, bc]
            if pd.notna(v):
                e0 = float(v)
                if e0 != 0:
                    eq = e0
                    break
        if ni is None or eq is None:
            return None
        return ni / abs(eq)
    except Exception as e:  # pragma: no cover
        _log.debug("roe_from_statements skip: %s", e)
        return None


def fetch_fundamentals_yfinance(
    symbols: list[str],
    *,
    delay_s: float = _DEFAULT_DELAY_S,
) -> pd.DataFrame:
    """
    One row per **NSE tradingsymbol** (not ``.NS`` suffix) with raw Yahoo fields.

    Raises ``ImportError`` if yfinance is not installed.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for fundamentals. Install with: pip install yfinance "
            "or pip install -e '.[brokers]'"
        ) from e

    uniq = sorted({str(s).strip().upper() for s in symbols if s and not str(s).startswith("^")})
    rows: list[dict[str, Any]] = []

    for i, sym in enumerate(uniq):
        ysym = _to_nse_yahoo_symbol(sym)
        row: dict[str, Any] = {"symbol": sym}
        for c in _RAW_COLS:
            row[c] = np.nan

        try:
            t = yf.Ticker(ysym)
            info = t.info or {}

            tp = info.get("trailingPE")
            row["yf_trailing_pe"] = float(tp) if tp is not None and pd.notna(tp) else np.nan

            fp = info.get("forwardPE")
            row["yf_forward_pe"] = float(fp) if fp is not None and pd.notna(fp) else np.nan

            pb = info.get("priceToBook")
            row["yf_pb"] = float(pb) if pb is not None and pd.notna(pb) else np.nan

            dy = info.get("dividendYield")
            if dy is not None and pd.notna(dy):
                row["yf_dividend_yield"] = float(dy) if float(dy) <= 1.0 else float(dy) / 100.0
            else:
                row["yf_dividend_yield"] = np.nan

            de = info.get("debtToEquity")
            row["yf_debt_to_equity"] = float(de) if de is not None and pd.notna(de) else np.nan

            cr = info.get("currentRatio")
            row["yf_current_ratio"] = float(cr) if cr is not None and pd.notna(cr) else np.nan

            roe = info.get("returnOnEquity")
            row["yf_roe"] = float(roe) if roe is not None and pd.notna(roe) else np.nan

            pm = info.get("profitMargins")
            row["yf_profit_margin"] = float(pm) if pm is not None and pd.notna(pm) else np.nan

            bm = _balance_sheet_metrics(t)
            crbs = bm.get("yf_current_ratio_bs")
            debs = bm.get("yf_debt_to_equity_bs")
            row["yf_current_ratio_bs"] = np.nan if crbs is None else float(crbs)
            row["yf_debt_to_equity_bs"] = np.nan if debs is None else float(debs)

            # NSE ``Ticker.info`` often omits currentRatio / returnOnEquity; backfill from statements.
            if pd.isna(row["yf_current_ratio"]) and crbs is not None:
                row["yf_current_ratio"] = float(crbs)
            if pd.isna(row["yf_roe"]):
                roe_fb = _roe_from_statements(t)
                if roe_fb is not None:
                    row["yf_roe"] = roe_fb
        except Exception as e:
            _log.warning("yfinance failed for %s (%s): %s", sym, ysym, e)

        rows.append(row)
        if delay_s > 0 and i + 1 < len(uniq):
            time.sleep(delay_s)

    return pd.DataFrame(rows).set_index("symbol")


def merge_fundamentals_into_features(
    feats: pd.DataFrame,
    fund: pd.DataFrame,
    sector_map: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Left-join fundamentals onto ``feats`` (MultiIndex date, symbol), add sector
    relatives where possible, then cross-sectional ``*_z`` columns.
    """
    if fund.empty:
        return feats

    fund = fund.copy()
    fund.index = fund.index.astype(str).str.upper()

    out = feats.copy()
    lvl1 = out.index.get_level_values(1).astype(str).str.upper()
    lvl0 = out.index.get_level_values(0)
    lvl1_list = lvl1.tolist()

    for c in _RAW_COLS:
        if c not in fund.columns:
            continue
        ser = fund[c]
        if ser.index.has_duplicates:
            ser = ser[~ser.index.duplicated(keep="last")]
        fd = ser.to_dict()
        out[c] = [fd.get(s, np.nan) for s in lvl1_list]

    if sector_map is not None and not sector_map.empty:
        sm = sector_map.copy()
        sm.index = sm.index.astype(str).str.upper()
        if sm.index.has_duplicates:
            sm = sm[~sm.index.duplicated(keep="last")]
        sdict = sm.to_dict()
        sec = np.array([sdict.get(s, "Others") for s in lvl1_list])

        def rel(col: str, outn: str) -> None:
            if col not in out.columns:
                return
            x = out[col].astype(float)
            df = pd.DataFrame({"x": x, "d": np.asarray(lvl0), "sec": sec})
            med = df.groupby(["d", "sec"], sort=False)["x"].transform("median")
            out[outn] = x - med

        rel("yf_trailing_pe", "yf_pe_vs_sector")
        rel("yf_pb", "yf_pb_vs_sector")
        rel("yf_dividend_yield", "yf_div_yield_vs_sector")

    z_cols = [
        "yf_trailing_pe",
        "yf_forward_pe",
        "yf_pb",
        "yf_dividend_yield",
        "yf_debt_to_equity",
        "yf_current_ratio",
        "yf_roe",
        "yf_profit_margin",
        "yf_current_ratio_bs",
        "yf_debt_to_equity_bs",
        "yf_pe_vs_sector",
        "yf_pb_vs_sector",
        "yf_div_yield_vs_sector",
    ]
    for c in z_cols:
        if c not in out.columns:
            continue
        out[f"{c}_z"] = out.groupby(level=0)[c].transform(_cs_zscore)

    # Keep raw ``yf_*`` and sector-spread columns for dashboards / rank JSON; trainers
    # still select only columns ending in ``_z``.
    n_ok = sum(1 for c in out.columns if c.startswith("yf_") and c.endswith("_z"))
    _log.info("Merged yfinance fundamentals: %d z-feature columns", n_ok)
    return out


def enrich_features_with_yfinance(
    feats: pd.DataFrame,
    *,
    sector_map: pd.Series | None = None,
    delay_s: float = _DEFAULT_DELAY_S,
) -> pd.DataFrame:
    """Fetch Yahoo fundamentals for symbols in ``feats`` and merge."""
    symbols = feats.index.get_level_values(1).unique().tolist()
    fund = fetch_fundamentals_yfinance(symbols, delay_s=delay_s)
    return merge_fundamentals_into_features(feats, fund, sector_map=sector_map)


def enrich_day_slice_with_yfinance(
    day: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    sector_map: pd.Series | None = None,
    delay_s: float = _DEFAULT_DELAY_S,
) -> pd.DataFrame:
    """Attach Yahoo fundamentals to a **single-date** feature slice (symbol index).

    Used by ``rank_for_date`` when ``--yfinance-fundamentals`` is set so the rank
    file and UI show ratios **without** rebuilding the full features parquet.
    Cross-sectional z-scores are taken **within this day's cohort** only.
    """
    if day.empty:
        return day
    asof_n = pd.to_datetime(asof).tz_localize(None).normalize()
    syms = day.index.astype(str)
    fund = fetch_fundamentals_yfinance(syms.tolist(), delay_s=delay_s)
    multi = day.copy()
    multi.index = pd.MultiIndex.from_arrays(
        [np.repeat(asof_n, len(multi)), syms.to_numpy()],
        names=["date", "symbol"],
    )
    merged = merge_fundamentals_into_features(multi, fund, sector_map=sector_map)
    out = merged.xs(asof_n, level=0)
    return out.reindex(day.index)
