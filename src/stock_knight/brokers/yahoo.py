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
    - This adapter assumes you provide symbols already in Yahoo format.
    """

    auto_adjust: bool = False


class YahooFinanceEODClient(BrokerEODClient):
    def __init__(self, cfg: YahooConfig | None = None) -> None:
        self.cfg = cfg or YahooConfig()

    def list_symbols(self) -> BrokerSymbols:
        raise NotImplementedError(
            "Yahoo Finance cannot reliably list NSE symbols. "
            "Pass `--symbols-file` to `ingest` with one symbol per line (e.g., RELIANCE.NS)."
        )

    def fetch_ohlcv(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install 'stock-knight[brokers]'") from e

        start = pd.to_datetime(start).tz_localize(None)
        end = pd.to_datetime(end).tz_localize(None)

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

