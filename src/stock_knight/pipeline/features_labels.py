from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from stock_knight.data.sectors import load_sector_map
from stock_knight.features.fundamentals_yfinance import enrich_features_with_yfinance
from stock_knight.features.price_features import compute_price_features
from stock_knight.labels import (
    Horizons,
    make_excess_returns,
    make_forward_returns,
    make_sector_residual_returns,
)
from stock_knight.logger import get_logger
from stock_knight.storage import DataPaths, read_df_parquet, write_df_parquet

_log = get_logger(__name__)

# Map broker-specific or broken benchmark names → verified Yahoo Finance ticker.
# Covers: Angel One names, Zerodha names, Upstox NSE_INDEX names, old broken Yahoo tickers.
_TICKER_ALIASES: dict[str, str] = {
    # Angel One SmartAPI instrument names
    "NIFTY":              "^NSEI",
    "NIFTY50":            "^NSEI",
    "NIFTY100":           "^CNX100",
    "NIFTY200":           "^CNX200",
    "BANKNIFTY":          "^NSEI",
    "MIDCPNIFTY":         "^NSMIDCP",
    "NIFTYMIDCAP150":     "^NSMIDCP",
    "NIFTY500":           "^NSEI",
    # Zerodha Kite Connect instrument names
    "NIFTY 50":           "^NSEI",
    "NIFTY 100":          "^CNX100",
    "NIFTY 200":          "^CNX200",
    "NIFTY BANK":         "^NSEI",
    "NIFTY MIDCAP 50":    "^NSMIDCP",
    "NIFTY MIDCAP 150":   "^NSMIDCP",
    "NIFTY 500":          "^NSEI",
    # Upstox NSE_INDEX|name format
    "NSE_INDEX|Nifty 50":         "^NSEI",
    "NSE_INDEX|Nifty 100":        "^CNX100",
    "NSE_INDEX|Nifty 200":        "^CNX200",
    "NSE_INDEX|Nifty Bank":       "^NSEI",
    "NSE_INDEX|Nifty Midcap 50":  "^NSMIDCP",
    "NSE_INDEX|NIFTY MIDCAP 150": "^NSMIDCP",
    "NSE_INDEX|Nifty 500":        "^NSEI",
    # Old/broken Yahoo Finance tickers (tested 404 as of 2026-05)
    "^CNXMIDCAP":    "^NSMIDCP",
    "^CNXMIDCAP100": "^NSMIDCP",
    "^NSMIDCP100":   "^NSMIDCP",
    "^NSMIDCP50":    "^NSMIDCP",
    "^CNXSMALLCAP":  "^NSEI",
    "^CNX500":       "^NSEI",
    "^NIFTYBANK":    "^NSEI",
    "^CNXIT":        "^NSEI",
}

# Verified working Yahoo Finance benchmark tickers (tested 2026-05)
_VALID_BENCHMARKS = ["^NSEI", "^CNX100", "^CNX200", "^NSMIDCP", "^BSESN"]


def _auto_fetch_benchmark(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Try to download benchmark OHLCV from Yahoo Finance and return in standard format."""
    try:
        import yfinance as yf  # type: ignore[import]
    except ImportError:
        return None
    try:
        raw = yf.download(symbol, start=start, end=end + pd.Timedelta(days=1),
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        # Flatten multi-level columns if needed
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"date": "date"})
        raw["symbol"] = symbol
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in raw.columns:
                raw[col] = raw.get("close", 0.0)
        return raw[["date", "symbol", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None


def build_features_and_labels(
    data_root: Path,
    ohlcv_parquet: Path,
    benchmark_symbol: str | None = None,
    horizons: Horizons = Horizons(),
    sectors_csv: Path | None = None,
    yfinance_fundamentals: bool = False,
    yfinance_delay_s: float = 0.2,
) -> tuple[Path, Path]:
    """
    Reads OHLCV parquet, writes:
      - features parquet (indexed by date,symbol)
      - labels parquet (indexed by date,symbol)

    The labels parquet always contains raw ``r_fwd_h`` columns. When
    ``benchmark_symbol`` is set, additional ``r_fwd_h_excess`` columns are
    written (relative to the benchmark). When ``sectors_csv`` is provided
    (default: ``data/sectors.csv``), additional ``r_fwd_h_sector_residual``
    columns are written — these are the recommended training target for
    rank-loss objectives (Phase 3a).

    When ``yfinance_fundamentals`` is True, enriches features with Yahoo Finance
    snapshot ratios (P/E, P/B, dividend yield, balance-sheet metrics) and
    sector-relative spreads. Requires ``yfinance`` and may take minutes for
    large universes (see ``yfinance_delay_s``).
    """
    paths = DataPaths(root=data_root)
    ohlcv = read_df_parquet(ohlcv_parquet)

    # Resolve broker-specific or broken benchmark names to their Yahoo Finance equivalent
    if benchmark_symbol and benchmark_symbol in _TICKER_ALIASES:
        corrected = _TICKER_ALIASES[benchmark_symbol]
        warnings.warn(
            f"Benchmark {benchmark_symbol!r} is a broker-specific name — "
            f"mapped to Yahoo Finance ticker {corrected!r} for excess-return computation.",
            UserWarning,
            stacklevel=2,
        )
        benchmark_symbol = corrected

    feats = compute_price_features(ohlcv)
    labels = make_forward_returns(ohlcv[["date", "symbol", "close"]], horizons=horizons)

    if benchmark_symbol:
        available_symbols = labels.index.get_level_values(1).unique().tolist()

        # Auto-fetch benchmark if it wasn't in the ingested data
        if benchmark_symbol not in available_symbols:
            ohlcv_dates = pd.to_datetime(ohlcv["date"])
            data_start = ohlcv_dates.min()
            data_end = ohlcv_dates.max()
            warnings.warn(
                f"Benchmark {benchmark_symbol!r} not in parquet — fetching from Yahoo Finance automatically...",
                UserWarning,
                stacklevel=2,
            )
            bench_ohlcv = _auto_fetch_benchmark(benchmark_symbol, data_start, data_end)
            if bench_ohlcv is not None and not bench_ohlcv.empty:
                # Merge benchmark rows into labels only (not features — we don't rank the index)
                bench_labels = make_forward_returns(
                    bench_ohlcv[["date", "symbol", "close"]], horizons=horizons
                )
                labels = pd.concat([labels, bench_labels])
                labels = labels.sort_index()
                available_symbols = labels.index.get_level_values(1).unique().tolist()
                _log.info("Auto-fetched benchmark %r: %d rows", benchmark_symbol, len(bench_ohlcv))
            else:
                raise ValueError(
                    f"benchmark_symbol={benchmark_symbol!r} was not found in the OHLCV parquet "
                    f"and could not be fetched from Yahoo Finance.\n"
                    f"Valid alternatives: {_VALID_BENCHMARKS}\n"
                    f"Run ingest with --benchmark-symbol {benchmark_symbol!r} first, or use "
                    f"--benchmark-symbol '^NSEI' (Nifty 50) which is always available."
                )

        if benchmark_symbol not in available_symbols:
            raise ValueError(f"benchmark_symbol={benchmark_symbol!r} has no data rows in labels")

        bench = labels.xs(benchmark_symbol, level=1, drop_level=False)
        if len(bench) == 0:
            raise ValueError(f"benchmark_symbol={benchmark_symbol!r} has no data rows in labels")

        bench_by_date = bench.reset_index().set_index("date").drop(columns=["symbol"])
        excess = make_excess_returns(labels, bench_by_date)
        labels = labels.join(excess, how="left")

        # Do not train on benchmark itself
        feats = feats[feats.index.get_level_values(1) != benchmark_symbol]
        labels = labels[labels.index.get_level_values(1) != benchmark_symbol]

    sector_map = load_sector_map(sectors_csv)

    if yfinance_fundamentals:
        feats = enrich_features_with_yfinance(
            feats,
            sector_map=sector_map if not sector_map.empty else None,
            delay_s=yfinance_delay_s,
        )

    # Sector-residual targets (Phase 3a). Computed across all symbols still
    # present at this point. Safe even when the sector map is empty —
    # everything just falls into the "Others" bucket.
    if not sector_map.empty:
        sector_residuals = make_sector_residual_returns(
            labels[[c for c in labels.columns if c.startswith("r_fwd_") and "_excess" not in c and "_sector_residual" not in c]],
            sector_map=sector_map,
        )
        labels = labels.join(sector_residuals, how="left")

    feats_out = paths.features / f"features_{ohlcv_parquet.stem}.parquet"
    labels_out = paths.labels / f"labels_{ohlcv_parquet.stem}.parquet"

    write_df_parquet(feats_out, feats)
    write_df_parquet(labels_out, labels)

    return feats_out, labels_out

