from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    Covers:
    - Historical OHLCV data  (fetch_ohlcv)
    - Live last traded price  (fetch_ltp)
    - Account info            (get_funds, get_holdings, get_positions)
    - Order management        (place_order, cancel_order, get_orders)
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

    def fetch_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Spot LTP from Kite `ltp()` for `EXCHANGE:TRADINGSYMBOL` instrument keys."""
        originals: list[str] = []
        for s in symbols:
            if not (s or "").strip():
                continue
            originals.append(str(s).strip())
        if not originals:
            return {}

        keys: list[str] = []
        exch = self.cfg.exchange.upper()
        for o in originals:
            ts = o.upper().strip()
            if ts.endswith(".NS"):
                ts = ts[: -len(".NS")].strip()
            keys.append(f"{exch}:{ts}")

        try:
            q = self._kite.ltp(keys)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Zerodha Kite LTP failed: {e}") from e

        out: dict[str, float] = {}
        for o, k in zip(originals, keys, strict=True):
            row = q.get(k)
            if row is None:
                continue
            lp = row.get("last_price")
            if lp is not None:
                out[o] = float(lp)
        return out

    # ------------------------------------------------------------------
    # Account / portfolio
    # ------------------------------------------------------------------

    def get_funds(self) -> dict[str, Any]:
        """
        Returns available cash and margin for the equity segment.

        Keys in response (float, INR):
          available_cash, used_margin, net, opening_balance
        """
        try:
            margins = self._kite.margins(segment="equity")
        except Exception as e:
            raise RuntimeError(f"Zerodha get_funds failed: {e}") from e
        avail = margins.get("net", {})
        return {
            "available_cash": float(avail.get("available", {}).get("cash", 0)),
            "opening_balance": float(avail.get("available", {}).get("opening_balance", 0)),
            "used_margin": float(avail.get("utilised", {}).get("debits", 0)),
            "net": float(avail.get("net", 0)),
        }

    def get_holdings(self) -> pd.DataFrame:
        """
        Current long-term equity holdings (Demat).

        Returns DataFrame with columns:
          tradingsymbol, quantity, average_price, last_price, pnl, pnl_pct
        """
        try:
            raw = self._kite.holdings()
        except Exception as e:
            raise RuntimeError(f"Zerodha get_holdings failed: {e}") from e

        rows = []
        for h in raw:
            qty = int(h.get("quantity", 0))
            if qty <= 0:
                continue
            avg = float(h.get("average_price", 0))
            lp = float(h.get("last_price", avg) or avg)
            pnl = (lp - avg) * qty
            pnl_pct = ((lp / avg) - 1.0) * 100.0 if avg > 0 else 0.0
            rows.append(
                {
                    "tradingsymbol": h.get("tradingsymbol"),
                    "exchange": h.get("exchange"),
                    "quantity": qty,
                    "average_price": avg,
                    "last_price": lp,
                    "current_value": lp * qty,
                    "pnl": pnl,
                    "pnl_pct": round(pnl_pct, 2),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["tradingsymbol", "exchange", "quantity", "average_price",
                     "last_price", "current_value", "pnl", "pnl_pct"]
        )

    def get_positions(self) -> pd.DataFrame:
        """
        Intraday and short-term positions (MIS / CNC open today).

        Returns DataFrame with columns:
          tradingsymbol, quantity, average_price, last_price, pnl, product
        """
        try:
            raw = self._kite.positions()
        except Exception as e:
            raise RuntimeError(f"Zerodha get_positions failed: {e}") from e

        rows = []
        for segment in ("day", "net"):
            for p in raw.get(segment, []):
                qty = int(p.get("quantity", 0))
                if qty == 0:
                    continue
                avg = float(p.get("average_price", 0))
                lp = float(p.get("last_price", avg) or avg)
                rows.append(
                    {
                        "tradingsymbol": p.get("tradingsymbol"),
                        "exchange": p.get("exchange"),
                        "product": p.get("product"),
                        "quantity": qty,
                        "average_price": avg,
                        "last_price": lp,
                        "pnl": float(p.get("pnl", 0)),
                        "segment": segment,
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["tradingsymbol", "exchange", "product", "quantity",
                     "average_price", "last_price", "pnl", "segment"]
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        tradingsymbol: str,
        transaction_type: str,  # "BUY" or "SELL"
        quantity: int,
        order_type: str = "MARKET",  # "MARKET" | "LIMIT"
        price: float | None = None,  # required for LIMIT
        product: str = "CNC",  # "CNC" (delivery) | "MIS" (intraday)
        exchange: str | None = None,
        tag: str = "stock-knight",
    ) -> str:
        """
        Place a Kite Connect equity order.

        Returns Kite order_id string.
        Raises RuntimeError on failure.
        """
        from kiteconnect import KiteConnect  # noqa: F401 (imported for constants)

        exch = (exchange or self.cfg.exchange).upper()
        ts = tradingsymbol.upper().strip()
        if ts.endswith(".NS"):
            ts = ts[: -len(".NS")]

        kwargs: dict[str, Any] = {
            "exchange": exch,
            "tradingsymbol": ts,
            "transaction_type": transaction_type.upper(),
            "quantity": int(quantity),
            "product": product.upper(),
            "order_type": order_type.upper(),
            "tag": tag,
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("price is required for LIMIT orders")
            kwargs["price"] = float(price)

        try:
            result = self._kite.place_order(variety="regular", **kwargs)
        except Exception as e:
            raise RuntimeError(f"Zerodha place_order failed for {ts}: {e}") from e

        return str(result)

    def get_orders(self) -> pd.DataFrame:
        """
        Today's orders (all statuses).

        Returns DataFrame with columns:
          order_id, tradingsymbol, transaction_type, status, quantity,
          price, average_price, order_type, product, placed_at
        """
        try:
            raw = self._kite.orders()
        except Exception as e:
            raise RuntimeError(f"Zerodha get_orders failed: {e}") from e

        rows = []
        for o in raw:
            rows.append(
                {
                    "order_id": o.get("order_id"),
                    "tradingsymbol": o.get("tradingsymbol"),
                    "exchange": o.get("exchange"),
                    "transaction_type": o.get("transaction_type"),
                    "status": o.get("status"),
                    "quantity": int(o.get("quantity", 0)),
                    "filled_quantity": int(o.get("filled_quantity", 0)),
                    "price": float(o.get("price", 0)),
                    "average_price": float(o.get("average_price", 0)),
                    "order_type": o.get("order_type"),
                    "product": o.get("product"),
                    "placed_at": str(o.get("order_timestamp", "")),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["order_id", "tradingsymbol", "exchange", "transaction_type",
                     "status", "quantity", "filled_quantity", "price",
                     "average_price", "order_type", "product", "placed_at"]
        )

    def cancel_order(self, order_id: str) -> str:
        """Cancel a pending Kite order. Returns order_id on success."""
        try:
            result = self._kite.cancel_order(variety="regular", order_id=order_id)
        except Exception as e:
            raise RuntimeError(f"Zerodha cancel_order {order_id} failed: {e}") from e
        return str(result)

