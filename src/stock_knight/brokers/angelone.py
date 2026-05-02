"""
Angel One SmartAPI broker adapter.

Covers:
- Session creation with auto-TOTP  (generateSession)
- Historical OHLCV data            (fetch_ohlcv)
- Live last traded price           (fetch_ltp)
- Account funds / margins          (get_funds)
- Demat holdings                   (get_holdings)
- Intraday positions               (get_positions)
- Order placement                  (place_order)
- Order log                        (get_orders)
- Order cancellation               (cancel_order)

Symbol token lookup:
  Angel One uses numeric symboltoken (e.g. "3045" for SBIN-EQ).
  A master instrument CSV is downloaded once per session to build a
  tradingsymbol → symboltoken map.

Free API: SmartAPI is free for Angel One trading account holders.
Docs: https://smartapi.angelbroking.com/docs
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any

from stock_knight.logger import get_logger

_log = get_logger(__name__)

import pandas as pd

from stock_knight.brokers.base import BrokerEODClient, BrokerSymbols


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AngelOneConfig:
    api_key: str
    client_code: str
    password: str
    totp_secret: str = ""   # Base-32 TOTP secret (recommended). If empty, totp_value is used.
    totp_value: str = ""    # One-time TOTP code (fallback; expires in 30s).
    exchange: str = "NSE"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AngelOneSmartApiClient(BrokerEODClient):
    """
    Full Angel One SmartAPI client (data + trading).

    Parameters
    ----------
    cfg : AngelOneConfig
        Credentials and exchange defaults.
    """

    def __init__(self, cfg: AngelOneConfig) -> None:
        self.cfg = cfg
        try:
            from SmartApi import SmartConnect
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: pip install 'stock-knight[brokers]'"
            ) from e

        totp_code = self._resolve_totp(cfg)
        self._smart = SmartConnect(api_key=cfg.api_key)
        session = self._smart.generateSession(cfg.client_code, cfg.password, totp_code)
        if not isinstance(session, dict) or session.get("status") is False:
            msg = (session or {}).get("message", "unknown error")
            raise RuntimeError(f"Angel One session failed: {msg}")

        # Store auth token for reference
        data = session.get("data") or {}
        self._auth_token: str = data.get("jwtToken", "")
        self._refresh_token: str = data.get("refreshToken", "")

        # Lazy-loaded instrument map: {exchange|tradingsymbol → symboltoken}
        self._token_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_totp(cfg: AngelOneConfig) -> str:
        """Return a live TOTP code from either the secret or an explicit value."""
        if cfg.totp_secret:
            try:
                import pyotp
            except ImportError as e:
                raise RuntimeError(
                    "pyotp is required for TOTP secret support: pip install pyotp"
                ) from e
            return pyotp.TOTP(cfg.totp_secret).now()
        if cfg.totp_value:
            return cfg.totp_value
        raise ValueError(
            "Provide either totp_secret (recommended) or totp_value in AngelOneConfig."
        )

    def _build_token_map(self) -> None:
        """
        Download Angel One's master instrument list and build a symbol → token map.
        Cached for the lifetime of the session.
        """
        if self._token_map:
            return
        try:
            import urllib.request
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            with urllib.request.urlopen(url, timeout=30) as r:
                import json
                instruments = json.loads(r.read())
        except Exception as e:
            raise RuntimeError(f"Failed to download Angel One instrument master: {e}") from e

        for row in instruments:
            exch = str(row.get("exch_seg", "")).upper()
            ts = str(row.get("symbol", "")).upper().split("-")[0].strip()
            token = str(row.get("token", ""))
            if ts and token:
                self._token_map[f"{exch}|{ts}"] = token
                self._token_map[f"{exch}|{ts}-EQ"] = token

    def _resolve_token(self, symbol: str, exchange: str | None = None) -> str | None:
        """Look up symboltoken for a tradingsymbol. Returns None if not found."""
        self._build_token_map()
        exch = (exchange or self.cfg.exchange).upper()
        ts = symbol.upper().strip()
        if ts.endswith(".NS"):
            ts = ts[: -len(".NS")]
        return (
            self._token_map.get(f"{exch}|{ts}")
            or self._token_map.get(f"{exch}|{ts}-EQ")
        )

    # ------------------------------------------------------------------
    # BrokerEODClient interface
    # ------------------------------------------------------------------

    def list_symbols(self) -> BrokerSymbols:
        self._build_token_map()
        exch = self.cfg.exchange.upper()
        syms = [
            k.split("|")[1]
            for k in self._token_map
            if k.startswith(f"{exch}|") and not k.endswith("-EQ")
        ]
        return BrokerSymbols(symbols=sorted(set(syms)))

    # Angel One SmartAPI rate limits: ~1 req/sec for historical candles.
    # We use a conservative 0.4s delay between symbols + exponential backoff on
    # rate-limit errors (up to 3 retries with 2s, 5s, 10s waits).
    _CANDLE_DELAY_SEC: float = 0.4
    _RATE_LIMIT_MSGS: tuple[str, ...] = (
        "access rate",
        "rate limit",
        "too many requests",
        "throttl",
    )

    def _candle_with_retry(self, params: dict, sym: str) -> list | None:
        """Call getCandleData with automatic retry on rate-limit errors."""
        delays = [2, 5, 10]
        for attempt, wait in enumerate([-1] + delays):
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self._smart.getCandleData(params)
                data = resp.get("data") if isinstance(resp, dict) else None
                return data or []
            except Exception as exc:
                msg = str(exc).lower()
                is_rate_limit = any(kw in msg for kw in self._RATE_LIMIT_MSGS)
                if is_rate_limit and attempt < len(delays):
                    _log.warning(
                        "AngelOne rate limit hit for %s — retrying in %ds (attempt %d/%d)",
                        sym, delays[attempt], attempt + 1, len(delays),
                    )
                    continue
                # Non-rate-limit error or retries exhausted
                _log.error("AngelOne getCandleData failed for %s: %s", sym, exc)
                return None
        return None

    def fetch_ohlcv(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        start = pd.to_datetime(start).tz_localize(None)
        end = pd.to_datetime(end).tz_localize(None)

        rows: list[dict] = []
        total = len(symbols)
        for idx, sym in enumerate(symbols, 1):
            token = self._resolve_token(sym)
            if token is None:
                continue
            params = {
                "exchange": self.cfg.exchange,
                "symboltoken": token,
                "interval": "ONE_DAY",
                "fromdate": start.strftime("%Y-%m-%d 09:15"),
                "todate": end.strftime("%Y-%m-%d 15:30"),
            }
            if idx % 20 == 0:
                _log.info("AngelOne OHLCV: fetched %d/%d symbols", idx, total)
            data = self._candle_with_retry(params, sym)
            if not data:
                time.sleep(self._CANDLE_DELAY_SEC)
                continue
            for c in data:
                # c: [timestamp, open, high, low, close, volume]
                rows.append(
                    {
                        "date": pd.Timestamp(c[0]).to_pydatetime(),
                        "symbol": sym,
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": int(c[5]) if len(c) > 5 else 0,
                    }
                )
            # Throttle: stay well under Angel One's rate limit
            time.sleep(self._CANDLE_DELAY_SEC)

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["date", "symbol", "open", "high", "low", "close", "volume"]
        )

    def fetch_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Live last traded price via SmartAPI `ltpData`."""
        out: dict[str, float] = {}
        for sym in symbols:
            token = self._resolve_token(sym)
            if token is None:
                continue
            try:
                ts = sym.upper().replace(".NS", "").strip()
                if not ts.endswith("-EQ"):
                    ts = ts + "-EQ"
                resp = self._smart.ltpData(
                    exchange=self.cfg.exchange,
                    tradingsymbol=ts,
                    symboltoken=token,
                )
                ltp = (resp.get("data") or {}).get("ltp")
                if ltp is not None:
                    out[sym] = float(ltp)
            except Exception:
                continue
            time.sleep(0.2)  # 5 req/sec max for LTP calls
        return out

    # ------------------------------------------------------------------
    # Account / portfolio
    # ------------------------------------------------------------------

    def get_funds(self) -> dict[str, Any]:
        """
        Available funds and margins.

        Keys (float, INR):
          available_cash, used_margin, net
        """
        try:
            resp = self._smart.rmsLimit()
        except Exception as e:
            raise RuntimeError(f"Angel One get_funds failed: {e}") from e

        data = (resp or {}).get("data") or {}
        net = float(data.get("net", 0) or 0)
        used = float(data.get("utilisedamount", 0) or 0)
        available = float(data.get("availablecash", 0) or 0)
        return {
            "available_cash": available,
            "used_margin": used,
            "net": net,
        }

    def get_holdings(self) -> pd.DataFrame:
        """
        Demat holdings with P&L.

        Columns: tradingsymbol, exchange, quantity, average_price,
                 last_price, current_value, pnl, pnl_pct
        """
        try:
            resp = self._smart.holding()
        except Exception as e:
            raise RuntimeError(f"Angel One get_holdings failed: {e}") from e

        raw = (resp or {}).get("data") or []
        rows = []
        for h in raw:
            qty = int(h.get("quantity", 0) or 0)
            if qty <= 0:
                continue
            avg = float(h.get("averageprice", 0) or 0)
            lp = float(h.get("ltp", avg) or avg)
            pnl = (lp - avg) * qty
            pnl_pct = ((lp / avg) - 1.0) * 100.0 if avg > 0 else 0.0
            rows.append(
                {
                    "tradingsymbol": h.get("tradingsymbol"),
                    "exchange": h.get("exchange"),
                    "symboltoken": h.get("symboltoken"),
                    "quantity": qty,
                    "average_price": avg,
                    "last_price": lp,
                    "current_value": lp * qty,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                }
            )
        cols = ["tradingsymbol", "exchange", "symboltoken", "quantity",
                "average_price", "last_price", "current_value", "pnl", "pnl_pct"]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    def get_positions(self) -> pd.DataFrame:
        """
        Open intraday / short-term positions.

        Columns: tradingsymbol, exchange, quantity, average_price,
                 last_price, pnl, product
        """
        try:
            resp = self._smart.position()
        except Exception as e:
            raise RuntimeError(f"Angel One get_positions failed: {e}") from e

        raw = (resp or {}).get("data") or []
        rows = []
        for p in raw:
            qty = int(p.get("netqty", 0) or 0)
            if qty == 0:
                continue
            avg = float(p.get("netprice", 0) or 0)
            lp = float(p.get("ltp", avg) or avg)
            rows.append(
                {
                    "tradingsymbol": p.get("tradingsymbol"),
                    "exchange": p.get("exchange"),
                    "product": p.get("producttype"),
                    "quantity": qty,
                    "average_price": avg,
                    "last_price": lp,
                    "pnl": float(p.get("pnl", 0) or 0),
                }
            )
        cols = ["tradingsymbol", "exchange", "product", "quantity",
                "average_price", "last_price", "pnl"]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(
        self,
        tradingsymbol: str,
        transaction_type: str,        # "BUY" | "SELL"
        quantity: int,
        order_type: str = "MARKET",   # "MARKET" | "LIMIT"
        price: float | None = None,   # required for LIMIT
        product: str = "DELIVERY",    # "DELIVERY" | "INTRADAY" | "CARRYFORWARD"
        exchange: str | None = None,
        tag: str = "stock-knight",
    ) -> str:
        """
        Place an Angel One equity order.

        Returns the order_id string on success.
        """
        exch = (exchange or self.cfg.exchange).upper()
        ts = tradingsymbol.upper().strip()
        if ts.endswith(".NS"):
            ts = ts[: -len(".NS")]
        # Angel One needs the full symbol name like SBIN-EQ
        ts_eq = ts if ts.endswith("-EQ") else f"{ts}-EQ"

        token = self._resolve_token(ts, exch)
        if token is None:
            raise RuntimeError(
                f"Symbol token not found for {ts}. "
                "Check that the symbol is a valid NSE equity name."
            )

        order_params: dict[str, Any] = {
            "variety": "NORMAL",
            "tradingsymbol": ts_eq,
            "symboltoken": token,
            "transactiontype": transaction_type.upper(),
            "exchange": exch,
            "ordertype": "MARKET" if order_type.upper() == "MARKET" else "LIMIT",
            "producttype": product.upper(),
            "duration": "DAY",
            "quantity": str(int(quantity)),
            "price": str(float(price)) if price is not None else "0",
            "triggerprice": "0",
            "squareoff": "0",
            "stoploss": "0",
        }

        try:
            resp = self._smart.placeOrder(order_params)
        except Exception as e:
            raise RuntimeError(f"Angel One place_order failed for {ts}: {e}") from e

        if not isinstance(resp, dict) or resp.get("status") is False:
            msg = (resp or {}).get("message", "unknown")
            raise RuntimeError(f"Angel One order rejected for {ts}: {msg}")

        return str((resp.get("data") or {}).get("orderid", ""))

    def get_orders(self) -> pd.DataFrame:
        """
        Today's order log.

        Columns: order_id, tradingsymbol, transaction_type, status,
                 quantity, filled_quantity, price, average_price,
                 order_type, product, placed_at
        """
        try:
            resp = self._smart.orderBook()
        except Exception as e:
            raise RuntimeError(f"Angel One get_orders failed: {e}") from e

        raw = (resp or {}).get("data") or []
        rows = []
        for o in raw:
            rows.append(
                {
                    "order_id": o.get("orderid"),
                    "tradingsymbol": o.get("tradingsymbol"),
                    "exchange": o.get("exchange"),
                    "transaction_type": o.get("transactiontype"),
                    "status": o.get("orderstatus"),
                    "quantity": int(o.get("quantity", 0) or 0),
                    "filled_quantity": int(o.get("filledshares", 0) or 0),
                    "price": float(o.get("price", 0) or 0),
                    "average_price": float(o.get("averageprice", 0) or 0),
                    "order_type": o.get("ordertype"),
                    "product": o.get("producttype"),
                    "placed_at": str(o.get("updatetime", "")),
                }
            )
        cols = ["order_id", "tradingsymbol", "exchange", "transaction_type",
                "status", "quantity", "filled_quantity", "price",
                "average_price", "order_type", "product", "placed_at"]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> str:
        """Cancel a pending Angel One order. Returns order_id on success."""
        try:
            resp = self._smart.cancelOrder(order_id, variety)
        except Exception as e:
            raise RuntimeError(f"Angel One cancel_order {order_id} failed: {e}") from e
        if not isinstance(resp, dict) or resp.get("status") is False:
            msg = (resp or {}).get("message", "unknown")
            raise RuntimeError(f"Cancel failed: {msg}")
        return str((resp.get("data") or {}).get("orderid", order_id))


# ---------------------------------------------------------------------------
# Backward-compat alias (old name used in cli.py)
# ---------------------------------------------------------------------------
AngelOneSmartApiEODClient = AngelOneSmartApiClient
