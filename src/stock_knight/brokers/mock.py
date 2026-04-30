from __future__ import annotations
 
import numpy as np
import pandas as pd
 
from stock_knight.brokers.base import BrokerEODClient, BrokerSymbols
 
 
class MockBrokerEODClient(BrokerEODClient):
     """
     Deterministic synthetic data generator so the full pipeline can run without
     live broker credentials during development.
     """
 
     def __init__(self, n_symbols: int = 150, seed: int = 7) -> None:
         self._n = n_symbols
         self._seed = seed
         self._symbols = [f"SYM{i:03d}" for i in range(1, n_symbols + 1)]
 
     def list_symbols(self) -> BrokerSymbols:
         return BrokerSymbols(symbols=self._symbols)
 
     def fetch_ohlcv(
         self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
     ) -> pd.DataFrame:
         start = pd.to_datetime(start).tz_localize(None)
         end = pd.to_datetime(end).tz_localize(None)
         dates = pd.bdate_range(start, end, freq="C")
 
         rng = np.random.default_rng(self._seed)
         rows: list[dict] = []
         for s in symbols:
             base = 100 + (hash(s) % 50)
             rets = rng.normal(0, 0.01, size=len(dates))
             price = base * np.exp(np.cumsum(rets))
             close = price
             open_ = close * (1 + rng.normal(0, 0.002, size=len(dates)))
             high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, size=len(dates)))
             low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, size=len(dates)))
             vol = rng.integers(50_000, 5_000_000, size=len(dates))
 
             for d, o, h, l, c, v in zip(dates, open_, high, low, close, vol, strict=False):
                 rows.append(
                     {
                         "date": pd.Timestamp(d).to_pydatetime(),
                         "symbol": s,
                         "open": float(o),
                         "high": float(h),
                         "low": float(l),
                         "close": float(c),
                         "volume": int(v),
                     }
                 )
 
         return pd.DataFrame(rows)
 
