from __future__ import annotations
 
from abc import ABC, abstractmethod
from dataclasses import dataclass
 
import pandas as pd
 
 
@dataclass(frozen=True)
class BrokerSymbols:
     symbols: list[str]
 
 
class BrokerEODClient(ABC):
     @abstractmethod
     def list_symbols(self) -> BrokerSymbols:
         """List tradable equity symbols (NSE)."""
 
     @abstractmethod
     def fetch_ohlcv(
         self,
         symbols: list[str],
         start: pd.Timestamp,
         end: pd.Timestamp,
     ) -> pd.DataFrame:
         """
         Return long-format OHLCV with columns:
           date, symbol, open, high, low, close, volume
         Dates should be trading-day timestamps (tz-naive is fine).
         """
 
