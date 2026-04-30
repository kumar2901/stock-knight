from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_knight.brokers.base import BrokerEODClient, BrokerSymbols


@dataclass(frozen=True)
class ZerodhaConfig:
    api_key: str
    access_token: str
    exchange: str = "NSE"


class ZerodhaKiteEODClient(BrokerEODClient):
    """
    Zerodha Kite Connect adapter.

    Important:
    - Kite historical data API uses instrument_token, not the tradingsymbol.
    - So you typically provide a symbols file of tradingsymbols and we resolve tokens.
    """

    def __init__(self, cfg: ZerodhaConfig) -> None:
        self.cfg = cfg
        try:
            from kiteconnect import KiteConnect
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install 'stock-knight[brokers]'") from e

        self._kite = KiteConnect(api_key=cfg.api_key)
        self._kite.set_access_token(cfg.access_token)

    def list_symbols(self) -> BrokerSymbols:
        # Listing all symbols can be large; prefer using --symbols-file.
        raise NotImplementedError(
            "Zerodha symbol listing is not enabled by default. "
            "Use `--symbols-file` with NSE tradingsymbols (e.g., RELIANCE)."
        )

    def fetch_ohlcv(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        start = pd.to_datetime(start).tz_localize(None)
        end = pd.to_datetime(end).tz_localize(None)

        # Resolve instrument tokens via instruments() once
        instruments = pd.DataFrame(self._kite.instruments(exchange=self.cfg.exchange))
        token_map = dict(
            zip(instruments["tradingsymbol"].astype(str), instruments["instrument_token"], strict=False)
        )

        rows: list[dict] = []
        for sym in symbols:
            token = token_map.get(sym)
            if token is None:
                continue
            candles = self._kite.historical_data(
                instrument_token=int(token),
                from_date=start.to_pydatetime(),
                to_date=end.to_pydatetime(),
                interval="day",
            )
            for c in candles:
                # c: {date, open, high, low, close, volume}
                rows.append(
                    {
                        "date": pd.Timestamp(c["date"]).to_pydatetime(),
                        "symbol": sym,
                        "open": float(c["open"]),
                        "high": float(c["high"]),
                        "low": float(c["low"]),
                        "close": float(c["close"]),
                        "volume": int(c.get("volume", 0)),
                    }
                )

        return pd.DataFrame(rows)

