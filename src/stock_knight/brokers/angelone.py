from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_knight.brokers.base import BrokerEODClient, BrokerSymbols


@dataclass(frozen=True)
class AngelOneConfig:
    api_key: str
    client_code: str
    password: str
    totp: str


class AngelOneSmartApiEODClient(BrokerEODClient):
    """
    Angel One SmartAPI adapter.

    Notes:
    - Package: `smartapi-python`
    - Historical candles require a symboltoken and exchange segment.
    - For production, you'll maintain a token mapping for your universe.
    """

    def __init__(self, cfg: AngelOneConfig) -> None:
        self.cfg = cfg
        try:
            from SmartApi import SmartConnect
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install 'stock-knight[brokers]'") from e

        self._smart = SmartConnect(api_key=cfg.api_key)
        session = self._smart.generateSession(cfg.client_code, cfg.password, cfg.totp)
        if not session or "data" not in session:
            raise RuntimeError("Angel One session generation failed")

    def list_symbols(self) -> BrokerSymbols:
        raise NotImplementedError(
            "Angel One symbol listing is not enabled by default. "
            "Use `--symbols-file` with symboltokens (or add a token mapping layer)."
        )

    def fetch_ohlcv(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        start = pd.to_datetime(start).tz_localize(None)
        end = pd.to_datetime(end).tz_localize(None)

        rows: list[dict] = []
        for symboltoken in symbols:
            params = {
                "exchange": "NSE",
                "symboltoken": symboltoken,
                "interval": "ONE_DAY",
                "fromdate": start.strftime("%Y-%m-%d 09:15"),
                "todate": end.strftime("%Y-%m-%d 15:30"),
            }
            resp = self._smart.getCandleData(params)
            data = resp.get("data") if isinstance(resp, dict) else None
            if not data:
                continue
            # Each candle: [timestamp, open, high, low, close, volume]
            for c in data:
                rows.append(
                    {
                        "date": pd.Timestamp(c[0]).to_pydatetime(),
                        "symbol": symboltoken,
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": int(c[5]) if len(c) > 5 else 0,
                    }
                )
        return pd.DataFrame(rows)

