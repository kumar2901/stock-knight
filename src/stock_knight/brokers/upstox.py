from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_knight.brokers.base import BrokerEODClient, BrokerSymbols


@dataclass(frozen=True)
class UpstoxConfig:
    access_token: str
    api_version: str = "2.0"


class UpstoxEODClient(BrokerEODClient):
    """
    Upstox adapter.

    Notes:
    - Upstox SDK package: `upstox-python-sdk`, imported as `upstox_client`.
    - Historic candles require instrument_key (not just tradingsymbol).
    - Provide `--symbols-file` with instrument_keys or a mapping file in your environment.
    """

    def __init__(self, cfg: UpstoxConfig) -> None:
        self.cfg = cfg
        try:
            import upstox_client
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install 'stock-knight[brokers]'") from e

        configuration = upstox_client.Configuration()
        configuration.access_token = cfg.access_token
        self._client = upstox_client.ApiClient(configuration)
        self._history_api = upstox_client.HistoryApi(self._client)

    def list_symbols(self) -> BrokerSymbols:
        raise NotImplementedError(
            "Upstox symbol listing is not enabled by default. "
            "Use `--symbols-file` with instrument_keys (preferred)."
        )

    def fetch_ohlcv(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        start = pd.to_datetime(start).tz_localize(None)
        end = pd.to_datetime(end).tz_localize(None)

        rows: list[dict] = []
        for instrument_key in symbols:
            # Upstox historic candles API returns arrays; SDK method names can differ by version.
            # We keep this as best-effort and raise a clear error if the SDK surface differs.
            try:
                resp = self._history_api.get_historical_candle_data1(
                    instrument_key=instrument_key,
                    interval="day",
                    from_date=start.date().isoformat(),
                    to_date=end.date().isoformat(),
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed calling Upstox history API. "
                    "You may need to update the SDK method name for your version."
                ) from e

            # Expected: resp.data.candles = [[ts, open, high, low, close, volume, ...], ...]
            candles = getattr(getattr(resp, "data", None), "candles", None)
            if candles is None:
                continue
            for c in candles:
                rows.append(
                    {
                        "date": pd.Timestamp(c[0]).to_pydatetime(),
                        "symbol": instrument_key,
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": int(c[5]) if len(c) > 5 else 0,
                    }
                )

        return pd.DataFrame(rows)

