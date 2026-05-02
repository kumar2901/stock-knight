"""
Upstox v2 broker adapter.

Covers:
- Instrument key resolution  (tradingsymbol → NSE_EQ|ISIN)
- Historical OHLCV data      (fetch_ohlcv)  — chunked into 1-year windows
- Live last traded price     (fetch_ltp)
- Account funds              (get_funds)
- Holdings                   (get_holdings)
- Positions                  (get_positions)
- Order management           (place_order, get_orders, cancel_order)

Instrument key format used by Upstox v2:  NSE_EQ|<ISIN>
e.g.  RELIANCE  →  NSE_EQ|INE002A01018

The master instrument list is downloaded once per session from:
  https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz
"""

from __future__ import annotations

import gzip
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from stock_knight.brokers.base import BrokerEODClient, BrokerSymbols


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpstoxConfig:
    access_token: str
    api_version: str = "2.0"
    exchange: str = "NSE_EQ"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class UpstoxEODClient(BrokerEODClient):
    """Upstox v2 full client (data + trading)."""

    _MASTER_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    _MAX_DAYS_PER_REQUEST = 365  # Upstox daily-candle limit per call

    def __init__(self, cfg: UpstoxConfig) -> None:
        self.cfg = cfg
        try:
            import upstox_client
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: pip install 'stock-knight[brokers]'") from e

        configuration = upstox_client.Configuration()
        configuration.access_token = cfg.access_token
        self._api_client = upstox_client.ApiClient(configuration)

        import upstox_client as uc
        self._history_api = uc.HistoryApi(self._api_client)
        self._market_api = uc.MarketQuoteApi(self._api_client)
        self._portfolio_api = uc.PortfolioApi(self._api_client)
        self._order_api = uc.OrderApi(self._api_client)
        self._user_api = uc.UserApi(self._api_client)

        # Lazy-loaded: tradingsymbol (upper) → instrument_key
        self._symbol_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Instrument key resolution
    # ------------------------------------------------------------------

    def _build_symbol_map(self) -> None:
        """Download NSE EQ instrument master and build symbol → instrument_key map."""
        if self._symbol_map:
            return
        try:
            with urllib.request.urlopen(self._MASTER_URL, timeout=30) as r:
                instruments = json.loads(gzip.decompress(r.read()))
        except Exception as e:
            raise RuntimeError(f"Failed to download Upstox instrument master: {e}") from e

        for row in instruments:
            if row.get("instrument_type") != "EQ":
                continue
            ts = str(row.get("trading_symbol", "")).upper().strip()
            ik = str(row.get("instrument_key", "")).strip()
            if ts and ik:
                self._symbol_map[ts] = ik

    def _resolve_key(self, symbol: str) -> str | None:
        """Return Upstox instrument_key for a plain tradingsymbol. Returns None if not found."""
        self._build_symbol_map()
        ts = symbol.upper().strip()
        if ts.endswith(".NS"):
            ts = ts[: -len(".NS")]
        # Direct match
        if ts in self._symbol_map:
            return self._symbol_map[ts]
        # Already a valid instrument key (contains |)
        if "|" in symbol:
            return symbol
        return None

    # ------------------------------------------------------------------
    # BrokerEODClient: symbol listing
    # ------------------------------------------------------------------

    def list_symbols(self) -> BrokerSymbols:
        self._build_symbol_map()
        return BrokerSymbols(symbols=sorted(self._symbol_map.keys()))

    # ------------------------------------------------------------------
    # BrokerEODClient: historical OHLCV
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        start = pd.to_datetime(start).tz_localize(None)
        end = pd.to_datetime(end).tz_localize(None)

        # Chunk date range into ≤365-day windows
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=self._MAX_DAYS_PER_REQUEST - 1), end)
            windows.append((chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)

        rows: list[dict] = []
        for sym in symbols:
            instrument_key = self._resolve_key(sym)
            if instrument_key is None:
                continue

            for w_start, w_end in windows:
                try:
                    resp = self._history_api.get_historical_candle_data1(
                        instrument_key=instrument_key,
                        interval="day",
                        to_date=w_end.date().isoformat(),
                        from_date=w_start.date().isoformat(),
                        api_version=self.cfg.api_version,
                    )
                except Exception as e:
                    # Skip individual windows that fail (e.g. no data for that period)
                    continue

                candles = getattr(getattr(resp, "data", None), "candles", None) or []
                for c in candles:
                    rows.append(
                        {
                            "date": pd.Timestamp(c[0]).tz_localize(None).to_pydatetime(),
                            "symbol": sym,
                            "open": float(c[1]),
                            "high": float(c[2]),
                            "low": float(c[3]),
                            "close": float(c[4]),
                            "volume": int(c[5]) if len(c) > 5 else 0,
                        }
                    )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["date", "symbol", "open", "high", "low", "close", "volume"]
        )

    # ------------------------------------------------------------------
    # Live LTP
    # ------------------------------------------------------------------

    def fetch_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Live last traded price via Upstox market quotes API."""
        keys: list[str] = []
        sym_to_key: dict[str, str] = {}
        for sym in symbols:
            k = self._resolve_key(sym)
            if k:
                keys.append(k)
                sym_to_key[sym] = k

        if not keys:
            return {}

        out: dict[str, float] = {}
        # Market quotes API accepts comma-separated instrument keys
        batch_size = 50
        key_to_sym = {v: k for k, v in sym_to_key.items()}
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            try:
                resp = self._market_api.ltp(
                    symbol=",".join(batch),
                    api_version=self.cfg.api_version,
                )
                data = (resp or {}).data or {}
                for ik, quote in (data or {}).items():
                    ltp = getattr(quote, "last_price", None)
                    sym = key_to_sym.get(ik)
                    if sym and ltp is not None:
                        out[sym] = float(ltp)
            except Exception:
                continue
        return out

    # ------------------------------------------------------------------
    # Account / portfolio
    # ------------------------------------------------------------------

    def get_funds(self) -> dict[str, Any]:
        """Available funds in equity segment."""
        try:
            resp = self._user_api.get_fund_and_margin(
                segment="SEC", api_version=self.cfg.api_version
            )
            data = (resp or {}).data or {}
            eq = getattr(data, "equity", data) or {}
            available = float(getattr(eq, "available_margin", 0) or 0)
            used = float(getattr(eq, "used_margin", 0) or 0)
            return {
                "available_cash": available,
                "used_margin": used,
                "net": available + used,
            }
        except Exception as e:
            raise RuntimeError(f"Upstox get_funds failed: {e}") from e

    def get_holdings(self) -> pd.DataFrame:
        """Demat holdings with P&L."""
        try:
            resp = self._portfolio_api.get_holdings(api_version=self.cfg.api_version)
        except Exception as e:
            raise RuntimeError(f"Upstox get_holdings failed: {e}") from e

        raw = getattr(getattr(resp, "data", None), "__iter__", None)
        items = list(resp.data) if resp and resp.data else []
        rows = []
        for h in items:
            qty = int(getattr(h, "quantity", 0) or 0)
            if qty <= 0:
                continue
            avg = float(getattr(h, "average_price", 0) or 0)
            lp = float(getattr(h, "last_price", avg) or avg)
            pnl = (lp - avg) * qty
            rows.append(
                {
                    "tradingsymbol": getattr(h, "trading_symbol", ""),
                    "instrument_key": getattr(h, "instrument_token", ""),
                    "quantity": qty,
                    "average_price": avg,
                    "last_price": lp,
                    "current_value": lp * qty,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(((lp / avg) - 1) * 100, 2) if avg > 0 else 0.0,
                }
            )
        cols = ["tradingsymbol", "instrument_key", "quantity", "average_price",
                "last_price", "current_value", "pnl", "pnl_pct"]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    def get_positions(self) -> pd.DataFrame:
        """Open intraday positions."""
        try:
            resp = self._portfolio_api.get_positions(api_version=self.cfg.api_version)
        except Exception as e:
            raise RuntimeError(f"Upstox get_positions failed: {e}") from e

        items = list(resp.data) if resp and resp.data else []
        rows = []
        for p in items:
            qty = int(getattr(p, "quantity", 0) or 0)
            if qty == 0:
                continue
            avg = float(getattr(p, "average_price", 0) or 0)
            lp = float(getattr(p, "last_price", avg) or avg)
            rows.append(
                {
                    "tradingsymbol": getattr(p, "trading_symbol", ""),
                    "product": getattr(p, "product", ""),
                    "quantity": qty,
                    "average_price": avg,
                    "last_price": lp,
                    "pnl": float(getattr(p, "pnl", 0) or 0),
                }
            )
        cols = ["tradingsymbol", "product", "quantity", "average_price", "last_price", "pnl"]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        tradingsymbol: str,
        transaction_type: str,       # "BUY" | "SELL"
        quantity: int,
        order_type: str = "MARKET",
        price: float | None = None,
        product: str = "D",          # "D" = delivery (CNC), "I" = intraday (MIS)
        exchange: str | None = None,
        tag: str = "stock-knight",
    ) -> str:
        """Place an Upstox equity order. Returns order_id."""
        import upstox_client as uc

        ts = tradingsymbol.upper().strip().replace(".NS", "")
        instrument_key = self._resolve_key(ts)
        if instrument_key is None:
            raise RuntimeError(
                f"Instrument key not found for {ts}. "
                "Check that it is a valid NSE equity tradingsymbol."
            )

        body = uc.PlaceOrderRequest(
            quantity=int(quantity),
            product=product.upper(),
            validity="DAY",
            price=float(price) if price is not None else 0,
            tag=tag,
            instrument_token=instrument_key,
            order_type=order_type.upper(),
            transaction_type=transaction_type.upper(),
            disclosed_quantity=0,
            trigger_price=0,
            is_amo=False,
        )
        try:
            resp = self._order_api.place_order(body, api_version=self.cfg.api_version)
        except Exception as e:
            raise RuntimeError(f"Upstox place_order failed for {ts}: {e}") from e

        data = getattr(resp, "data", None)
        return str(getattr(data, "order_id", "") if data else "")

    def get_orders(self) -> pd.DataFrame:
        """Today's order log."""
        try:
            resp = self._order_api.get_order_book(api_version=self.cfg.api_version)
        except Exception as e:
            raise RuntimeError(f"Upstox get_orders failed: {e}") from e

        items = list(resp.data) if resp and resp.data else []
        rows = []
        for o in items:
            rows.append(
                {
                    "order_id": getattr(o, "order_id", ""),
                    "tradingsymbol": getattr(o, "trading_symbol", ""),
                    "transaction_type": getattr(o, "transaction_type", ""),
                    "status": getattr(o, "status", ""),
                    "quantity": int(getattr(o, "quantity", 0) or 0),
                    "filled_quantity": int(getattr(o, "filled_quantity", 0) or 0),
                    "price": float(getattr(o, "price", 0) or 0),
                    "average_price": float(getattr(o, "average_price", 0) or 0),
                    "order_type": getattr(o, "order_type", ""),
                    "product": getattr(o, "product", ""),
                    "placed_at": str(getattr(o, "order_timestamp", "")),
                }
            )
        cols = ["order_id", "tradingsymbol", "transaction_type", "status", "quantity",
                "filled_quantity", "price", "average_price", "order_type", "product", "placed_at"]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    def cancel_order(self, order_id: str) -> str:
        """Cancel a pending Upstox order."""
        try:
            resp = self._order_api.cancel_order(
                order_id=order_id, api_version=self.cfg.api_version
            )
        except Exception as e:
            raise RuntimeError(f"Upstox cancel_order {order_id} failed: {e}") from e
        data = getattr(resp, "data", None)
        return str(getattr(data, "order_id", order_id) if data else order_id)
