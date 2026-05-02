from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_knight.brokers.base import BrokerEODClient, BrokerSymbols


@dataclass(frozen=True)
class YahooConfig:
    """
    Yahoo Finance via yfinance.

    Notes for NSE:
    - yfinance usually needs NSE tickers with suffix `.NS` (e.g., RELIANCE.NS).
    - By default we auto-normalize plain NSE symbols like `RELIANCE` to `RELIANCE.NS`.
    """

    auto_adjust: bool = False
    default_suffix: str = ".NS"


class YahooFinanceEODClient(BrokerEODClient):
    def __init__(self, cfg: YahooConfig | None = None) -> None:
        self.cfg = cfg or YahooConfig()

    def list_symbols(self) -> BrokerSymbols:
        raise NotImplementedError(
            "Yahoo Finance cannot reliably list NSE symbols. "
            "Pass `--symbols-file` to `ingest` with one symbol per line (e.g., RELIANCE or RELIANCE.NS)."
        )

    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        out: list[str] = []
        for s in symbols:
            s2 = (s or "").strip()
            if not s2:
                continue
            s2 = s2.upper()
            # Index / special tickers (e.g., ^NSEI) should be passed through
            if s2.startswith("^"):
                out.append(s2)
                continue
            if "." not in s2:
                s2 = s2 + self.cfg.default_suffix
            out.append(s2)
        return out

    def fetch_ohlcv(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install 'stock-knight[brokers]'") from e

        start = pd.to_datetime(start).tz_localize(None)
        end = pd.to_datetime(end).tz_localize(None)

        symbols = self._normalize_symbols(symbols)

        # yfinance download returns a wide panel for multiple tickers
        data = yf.download(
            tickers=" ".join(symbols),
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            group_by="ticker",
            auto_adjust=self.cfg.auto_adjust,
            threads=True,
            progress=False,
        )

        rows: list[dict] = []
        if isinstance(data.columns, pd.MultiIndex):
            # Multiple tickers
            for sym in symbols:
                if sym not in data.columns.get_level_values(0):
                    continue
                d = data[sym].dropna()
                for dt, r in d.iterrows():
                    rows.append(
                        {
                            "date": pd.Timestamp(dt).to_pydatetime(),
                            "symbol": sym,
                            "open": float(r.get("Open")),
                            "high": float(r.get("High")),
                            "low": float(r.get("Low")),
                            "close": float(r.get("Close")),
                            "volume": int(r.get("Volume", 0)),
                        }
                    )
        else:
            # Single ticker
            d = data.dropna()
            sym = symbols[0] if symbols else "UNKNOWN"
            for dt, r in d.iterrows():
                rows.append(
                    {
                        "date": pd.Timestamp(dt).to_pydatetime(),
                        "symbol": sym,
                        "open": float(r.get("Open")),
                        "high": float(r.get("High")),
                        "low": float(r.get("Low")),
                        "close": float(r.get("Close")),
                        "volume": int(r.get("Volume", 0)),
                    }
                )

        return pd.DataFrame(rows)

    def fetch_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Nearest live last price via recent daily closes (same symbol keys as inputs)."""
        try:
            import yfinance as yf
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install 'stock-knight[brokers]'") from e

        originals: list[str] = []
        for s in symbols:
            if not (s or "").strip():
                continue
            originals.append(str(s).strip())
        if not originals:
            return {}

        normalized = self._normalize_symbols(originals)

        uniq: list[str] = []
        seen: set[str] = set()
        for n in normalized:
            if n not in seen:
                seen.add(n)
                uniq.append(n)

        data = yf.download(
            tickers=" ".join(uniq),
            period="7d",
            group_by="ticker",
            auto_adjust=self.cfg.auto_adjust,
            threads=True,
            progress=False,
        )

        last_close: dict[str, float] = {}
        if len(uniq) == 1:
            sym = uniq[0]
            d = data.dropna()
            if not d.empty and "Close" in d.columns:
                last_close[sym] = float(d["Close"].iloc[-1])
        elif isinstance(data.columns, pd.MultiIndex):
            for sym in uniq:
                if sym not in data.columns.get_level_values(0):
                    continue
                d = data[sym].dropna()
                if d.empty:
                    continue
                last_close[sym] = float(d["Close"].iloc[-1])

        out: dict[str, float] = {}
        for orig, n in zip(originals, normalized, strict=True):
            if n in last_close:
                out[orig] = last_close[n]
        return out

