"""
Angel One trade execution module.

Flow:
  1. Load a rank JSON (produced by `stock-knight rank`).
  2. Fetch live LTP from Angel One SmartAPI for ranked symbols.
  3. Fetch current Demat holdings.
  4. Decide target portfolio: equal-weight long-only top-K within available capital.
  5. Diff target vs current → BUY orders for new entries, SELL orders for exits.
  6. Preview (dry-run) or execute via SmartAPI.

Usage from CLI:
  stock-knight execute plan \
    --broker angelone \
    --rank-file reports/ranks_7d_2025-12-30.json \
    --top-k 10 \
    --capital 100000 \
    --creds-file credentials.yaml

  stock-knight execute execute \
    --broker angelone \
    --rank-file reports/ranks_7d_2025-12-30.json \
    --top-k 10 \
    --capital 100000 \
    --creds-file credentials.yaml \
    --live                          # omit for dry-run
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

# Reuse the shared data structures from zerodha_trader
from stock_knight.execution.zerodha_trader import ExecutionPlan, OrderSpec

if TYPE_CHECKING:
    from stock_knight.brokers.angelone import AngelOneSmartApiClient


class AngelOneTrader:
    """
    Builds and (optionally) executes a top-K equity basket via Angel One SmartAPI.

    Parameters
    ----------
    client : AngelOneSmartApiClient
        Authenticated Angel One client.
    product : str
        "DELIVERY" for delivery (holds overnight), "INTRADAY" for same-day.
    order_type : str
        "MARKET" or "LIMIT". For LIMIT the LTP is used as the limit price.
    """

    def __init__(
        self,
        client: AngelOneSmartApiClient,
        product: str = "DELIVERY",
        order_type: str = "MARKET",
    ) -> None:
        self._client = client
        self.product = product.upper()
        self.order_type = order_type.upper()

    # ------------------------------------------------------------------
    # Passthrough account methods
    # ------------------------------------------------------------------

    def get_funds(self) -> dict:
        return self._client.get_funds()

    def get_holdings(self) -> pd.DataFrame:
        return self._client.get_holdings()

    def get_positions(self) -> pd.DataFrame:
        return self._client.get_positions()

    def get_orders(self) -> pd.DataFrame:
        return self._client.get_orders()

    # ------------------------------------------------------------------
    # Plan & execute
    # ------------------------------------------------------------------

    def build_plan(
        self,
        rank_file: Path | str,
        top_k: int = 10,
        capital: float | None = None,
        sell_exits: bool = True,
        min_order_value: float = 500.0,
    ) -> ExecutionPlan:
        """
        Build an execution plan from a rank JSON file.

        Parameters
        ----------
        rank_file     : path to rank JSON produced by `stock-knight rank`
        top_k         : number of top-ranked stocks to target
        capital       : INR to allocate; if None, uses full available cash
        sell_exits    : if True, sell existing holdings not in new top-K
        min_order_value : skip any order below this threshold (avoids tiny lots)
        """
        payload = json.loads(Path(rank_file).read_text())
        ranked: list[str] = payload.get("ranked") or []
        rows_raw: list[dict] = payload.get("rows") or []

        # LTP from rank file (live_ltp preferred, else asof_close)
        ltp_from_file: dict[str, float] = {}
        for r in rows_raw:
            sym = r.get("symbol", "")
            ltp = r.get("live_ltp") or r.get("asof_close")
            if sym and ltp:
                ltp_from_file[sym] = float(ltp)

        target_symbols = [s for s in ranked[:top_k] if s]

        # Fetch fresh LTP from Angel One for target symbols
        fresh_ltp: dict[str, float] = {}
        if target_symbols:
            try:
                fresh_ltp = self._client.fetch_ltp(target_symbols)
            except Exception:
                pass

        ltp: dict[str, float] = {**ltp_from_file, **fresh_ltp}

        # Account state
        funds = self._client.get_funds()
        available_cash = funds.get("available_cash", 0.0)
        capital_to_use = float(capital) if capital is not None else available_cash

        holdings_df = self._client.get_holdings()
        current_holdings: dict[str, int] = {}
        if not holdings_df.empty:
            for _, row in holdings_df.iterrows():
                ts = str(row["tradingsymbol"]).upper().replace("-EQ", "").strip()
                current_holdings[ts] = int(row["quantity"])

        plan = ExecutionPlan(
            rank_file=str(rank_file),
            horizon=payload.get("horizon", ""),
            asof=payload.get("asof", ""),
            top_k=top_k,
            capital_allocated=capital_to_use,
            available_cash=available_cash,
        )

        # BUY orders for top-K targets
        per_stock_capital = capital_to_use / len(target_symbols) if target_symbols else 0.0
        for sym in target_symbols:
            ts = sym.upper().replace(".NS", "").replace("-EQ", "").strip()
            price = ltp.get(sym) or ltp.get(ts)
            if not price or price <= 0:
                plan.skipped.append({"symbol": sym, "reason": "no LTP available"})
                continue
            already_held = current_holdings.get(ts, 0)
            desired_qty = math.floor(per_stock_capital / price)
            buy_qty = max(0, desired_qty - already_held)
            if buy_qty <= 0:
                plan.skipped.append({"symbol": sym, "reason": "already fully allocated or qty=0"})
                continue
            est_value = buy_qty * price
            if est_value < min_order_value:
                plan.skipped.append(
                    {"symbol": sym, "reason": f"order value ₹{est_value:.0f} below minimum ₹{min_order_value}"}
                )
                continue
            plan.orders.append(
                OrderSpec(
                    tradingsymbol=ts,
                    transaction_type="BUY",
                    quantity=buy_qty,
                    ltp=price,
                    estimated_value=round(est_value, 2),
                    order_type=self.order_type,
                    product=self.product,
                )
            )

        # SELL orders for holdings no longer in top-K
        if sell_exits and not holdings_df.empty:
            target_ts_set = {s.upper().replace(".NS", "").replace("-EQ", "").strip() for s in target_symbols}
            for _, row in holdings_df.iterrows():
                ts = str(row["tradingsymbol"]).upper().replace("-EQ", "").strip()
                qty = int(row["quantity"])
                if qty <= 0 or ts in target_ts_set:
                    continue
                price = float(row.get("last_price", 0) or 0)
                if price <= 0:
                    plan.skipped.append({"symbol": ts, "reason": "no exit LTP"})
                    continue
                plan.orders.append(
                    OrderSpec(
                        tradingsymbol=ts,
                        transaction_type="SELL",
                        quantity=qty,
                        ltp=price,
                        estimated_value=round(qty * price, 2),
                        order_type=self.order_type,
                        product=self.product,
                    )
                )

        return plan

    def execute(
        self,
        plan: ExecutionPlan,
        dry_run: bool = True,
    ) -> ExecutionPlan:
        """
        Execute orders in the plan via Angel One SmartAPI.

        Parameters
        ----------
        plan    : ExecutionPlan from build_plan()
        dry_run : if True, marks orders DRY_RUN without calling SmartAPI
        """
        for order in plan.orders:
            if dry_run:
                order.status = "DRY_RUN"
                continue
            try:
                order_id = self._client.place_order(
                    tradingsymbol=order.tradingsymbol,
                    transaction_type=order.transaction_type,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    price=order.ltp if order.order_type == "LIMIT" else None,
                    product=order.product,
                )
                order.order_id = order_id
                order.status = "PLACED"
            except Exception as e:
                order.status = "FAILED"
                order.error = str(e)
        return plan
