from __future__ import annotations
 
import argparse
from pathlib import Path
 
import pandas as pd
from rich.console import Console
 
from stock_knight.config import ProjectConfig, default_config
from stock_knight.data.corporate_actions import apply_price_adjustments
from stock_knight.data.index_universe import load_index_members
from stock_knight.data.qa import qa_ohlcv_frame
from stock_knight.data.universe import universe_by_liquidity_rule
from stock_knight.pipeline.features_labels import build_features_and_labels
from stock_knight.storage import DataPaths, ensure_dirs, read_df_parquet, write_df_parquet, write_json
from stock_knight.backtest.simulate import run_backtest
from stock_knight.backtest.portfolio import PortfolioSpec
from stock_knight.ranking.daily import rank_for_date, write_ranklists
from stock_knight.monitoring.drift import drift_report_from_scores, write_drift_report
from stock_knight.train.fit import train_walk_forward
from stock_knight.train.models import load_model
from stock_knight.credentials import load_credentials
 
 
def _get_broker_client(name: str, creds: dict | None = None):
    name = name.lower()
    creds = creds or {}
    if name == "mock":
        from stock_knight.brokers.mock import MockBrokerEODClient

        return MockBrokerEODClient()
    if name in {"yahoo", "yahoofinance"}:
        from stock_knight.brokers.yahoo import YahooFinanceEODClient

        return YahooFinanceEODClient()
    if name in {"zerodha", "kite"}:
        import os

        from stock_knight.brokers.zerodha import ZerodhaConfig, ZerodhaKiteEODClient

        z = creds.get("zerodha", {}) if isinstance(creds, dict) else {}
        api_key = z.get("api_key") or os.environ.get("ZERODHA_API_KEY")
        access_token = z.get("access_token") or os.environ.get("ZERODHA_ACCESS_TOKEN")
        if not api_key or not access_token:
            raise ValueError(
                "Missing Zerodha credentials. Provide --creds-file with zerodha.api_key/access_token "
                "or set env vars: ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN."
            )
        return ZerodhaKiteEODClient(ZerodhaConfig(api_key=api_key, access_token=access_token))
    if name == "upstox":
        import os

        from stock_knight.brokers.upstox import UpstoxConfig, UpstoxEODClient

        u = creds.get("upstox", {}) if isinstance(creds, dict) else {}
        access_token = u.get("access_token") or os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not access_token:
            raise ValueError(
                "Missing Upstox credentials. Provide --creds-file with upstox.access_token "
                "or set env var: UPSTOX_ACCESS_TOKEN."
            )
        api_version = u.get("api_version") or "2.0"
        return UpstoxEODClient(UpstoxConfig(access_token=access_token, api_version=api_version))
    if name in {"angelone", "angel"}:
        import os

        from stock_knight.brokers.angelone import AngelOneConfig, AngelOneSmartApiClient

        a = creds.get("angelone", {}) if isinstance(creds, dict) else {}
        api_key = a.get("api_key") or os.environ.get("ANGELONE_API_KEY")
        client_code = a.get("client_code") or os.environ.get("ANGELONE_CLIENT_CODE")
        password = a.get("password") or os.environ.get("ANGELONE_PASSWORD")
        totp_secret = a.get("totp_secret") or os.environ.get("ANGELONE_TOTP_SECRET", "")
        totp_value = a.get("totp_value") or a.get("totp") or os.environ.get("ANGELONE_TOTP", "")
        if not (api_key and client_code and password and (totp_secret or totp_value)):
            raise ValueError(
                "Missing Angel One credentials. Provide --creds-file with angelone.* "
                "(api_key, client_code, password, totp_secret or totp_value) "
                "or set env vars: ANGELONE_API_KEY, ANGELONE_CLIENT_CODE, ANGELONE_PASSWORD, ANGELONE_TOTP_SECRET."
            )
        return AngelOneSmartApiClient(
            AngelOneConfig(
                api_key=api_key,
                client_code=client_code,
                password=password,
                totp_secret=totp_secret,
                totp_value=totp_value,
            )
        )
    raise ValueError(f"Unknown broker={name}. Use one of: mock, yahoo, zerodha, upstox, angelone.")
 
 
def _make_trader(broker_name: str, broker_client, product: str, order_type: str):
    """Return the right trader class for the broker."""
    name = broker_name.lower()
    if name in {"angelone", "angel"}:
        from stock_knight.execution.angelone_trader import AngelOneTrader
        return AngelOneTrader(client=broker_client, product=product, order_type=order_type)
    # default: Zerodha
    from stock_knight.execution.zerodha_trader import ZerodhaTrader
    return ZerodhaTrader(client=broker_client, product=product, order_type=order_type)


def cmd_execute(args: argparse.Namespace) -> None:
    from rich.table import Table

    console = Console()

    creds = load_credentials(Path(args.creds_file)) if getattr(args, "creds_file", None) and args.creds_file else {}
    broker = _get_broker_client(args.broker, creds=creds)

    if not hasattr(broker, "place_order"):
        raise SystemExit(
            f"Broker '{args.broker}' does not support order execution. "
            "Use --broker zerodha or --broker angelone."
        )

    trader = _make_trader(args.broker, broker, args.product, args.order_type)

    dry_run: bool = not args.live

    if args.action == "funds":
        funds = trader.get_funds()
        console.print("[bold]Equity funds / margins[/bold]")
        for k, v in funds.items():
            console.print(f"  {k:<22}: ₹{v:,.2f}")
        return

    if args.action == "holdings":
        df = trader.get_holdings()
        if df.empty:
            console.print("[yellow]No holdings found.[/yellow]")
            return
        t = Table(title="Current holdings")
        for col in df.columns:
            t.add_column(str(col))
        for _, row in df.iterrows():
            t.add_row(*[str(v) for v in row.values])
        console.print(t)
        return

    if args.action == "orders":
        df = trader.get_orders()
        if df.empty:
            console.print("[yellow]No orders today.[/yellow]")
            return
        t = Table(title="Today's orders")
        for col in df.columns:
            t.add_column(str(col))
        for _, row in df.iterrows():
            t.add_row(*[str(v) for v in row.values])
        console.print(t)
        return

    if args.action == "plan":
        rank_file = Path(args.rank_file)
        if not rank_file.exists():
            raise SystemExit(f"Rank file not found: {rank_file}")
        plan = trader.build_plan(
            rank_file=rank_file,
            top_k=int(args.top_k),
            capital=float(args.capital) if args.capital else None,
            sell_exits=not args.no_sell_exits,
        )
        _print_plan(console, plan, dry_run=dry_run)
        if args.out:
            import json
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(plan.to_dict(), indent=2))
            console.print(f"\nSaved plan to {args.out}")
        return

    if args.action == "execute":
        rank_file = Path(args.rank_file)
        if not rank_file.exists():
            raise SystemExit(f"Rank file not found: {rank_file}")
        plan = trader.build_plan(
            rank_file=rank_file,
            top_k=int(args.top_k),
            capital=float(args.capital) if args.capital else None,
            sell_exits=not args.no_sell_exits,
        )
        _print_plan(console, plan, dry_run=dry_run)
        if not args.live:
            console.print(
                "\n[yellow]DRY RUN — no orders placed. Re-run with --live to execute.[/yellow]"
            )
        else:
            console.print(
                f"\n[bold red]LIVE EXECUTION[/bold red] — placing orders via {args.broker}..."
            )
        plan = trader.execute(plan, dry_run=dry_run)
        if args.live:
            _print_results(console, plan)
        if args.out:
            import json
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(plan.to_dict(), indent=2))
            console.print(f"Saved execution record to {args.out}")


def _print_plan(console: Console, plan, dry_run: bool) -> None:
    from rich.table import Table

    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]LIVE[/red]"
    console.print(
        f"\nExecution plan  {mode}  |  horizon={plan.horizon}  asof={plan.asof}  "
        f"top_k={plan.top_k}"
    )
    console.print(
        f"  Available cash   : ₹{plan.available_cash:,.2f}\n"
        f"  Capital allocated: ₹{plan.capital_allocated:,.2f}\n"
        f"  Total BUY value  : ₹{plan.total_buy_value:,.2f}\n"
        f"  Total SELL value : ₹{plan.total_sell_value:,.2f}"
    )
    if plan.orders:
        t = Table(title=f"Orders ({len(plan.orders)})", show_lines=True)
        t.add_column("Symbol")
        t.add_column("Side")
        t.add_column("Qty", justify="right")
        t.add_column("LTP ₹", justify="right")
        t.add_column("Est. value ₹", justify="right")
        t.add_column("Type")
        t.add_column("Product")
        for o in plan.orders:
            side_style = "green" if o.transaction_type == "BUY" else "red"
            t.add_row(
                o.tradingsymbol,
                f"[{side_style}]{o.transaction_type}[/{side_style}]",
                str(o.quantity),
                f"{o.ltp:,.2f}",
                f"{o.estimated_value:,.2f}",
                o.order_type,
                o.product,
            )
        console.print(t)
    if plan.skipped:
        console.print(f"[dim]Skipped {len(plan.skipped)} symbol(s):[/dim]")
        for s in plan.skipped:
            console.print(f"  [dim]{s.get('symbol')} — {s.get('reason')}[/dim]")


def _print_results(console: Console, plan) -> None:
    from rich.table import Table

    t = Table(title="Execution results", show_lines=True)
    t.add_column("Symbol")
    t.add_column("Side")
    t.add_column("Qty", justify="right")
    t.add_column("Status")
    t.add_column("Order ID")
    t.add_column("Error")
    for o in plan.orders:
        side_style = "green" if o.transaction_type == "BUY" else "red"
        status_style = "green" if o.status == "PLACED" else ("red" if o.status == "FAILED" else "yellow")
        t.add_row(
            o.tradingsymbol,
            f"[{side_style}]{o.transaction_type}[/{side_style}]",
            str(o.quantity),
            f"[{status_style}]{o.status}[/{status_style}]",
            o.order_id or "",
            o.error or "",
        )
    console.print(t)


def cmd_rank(args: argparse.Namespace) -> None:
    universe_name = (getattr(args, "universe", "") or "").lower().strip()
    universe: list[str] | None = None
    if universe_name and universe_name != "liquidity":
        universe = load_index_members(universe_name)

    fetch_ltp_cb = None
    ltp_meta: str | None = None
    if getattr(args, "live_ltp", False):
        b = str(args.ltp_broker or "").lower()
        if not b:
            raise SystemExit("--live-ltp requires --ltp-broker yahoo|zerodha|angelone|upstox")
        creds = load_credentials(Path(args.creds_file)) if getattr(args, "creds_file", None) and args.creds_file else {}
        bc = _get_broker_client(b, creds=creds)
        fetch_ltp_cb = bc.fetch_ltp
        if b in {"yahoo", "yahoofinance"}:
            ltp_meta = "yahoo"
        elif b in {"zerodha", "kite"}:
            ltp_meta = "zerodha"
        elif b in {"angelone", "angel"}:
            ltp_meta = "angelone"
        elif b == "upstox":
            ltp_meta = "upstox"

    ranks = rank_for_date(
        features_parquet=Path(args.features_parquet),
        ohlcv_parquet=Path(args.ohlcv_parquet) if args.ohlcv_parquet else None,
        model_path=Path(args.model_path),
        asof=pd.Timestamp(args.asof),
        horizon=str(args.horizon),
        training_target=str(args.training_target),
        benchmark_symbol=str(args.benchmark_symbol) if args.benchmark_symbol else None,
        fetch_ltp=fetch_ltp_cb,
        live_ltp_broker=ltp_meta,
        universe=universe,
        rank_method=str(getattr(args, "rank_method", "composite")),
    )
    write_ranklists(Path(args.out), ranks)


def cmd_ingest(args: argparse.Namespace) -> None:
    console = Console()
    cfg: ProjectConfig = default_config()
    root = Path(args.data_root)
    paths = DataPaths(root=root)
    ensure_dirs(paths.ohlcv, paths.corporate_actions, paths.universe)

    creds = load_credentials(args.creds_file)
    broker = _get_broker_client(args.broker, creds=creds)
    universe_name = (getattr(args, "universe", "") or "").lower().strip()
    if args.symbols_file:
        symbols = [s.strip() for s in Path(args.symbols_file).read_text().splitlines() if s.strip()]
    elif universe_name and universe_name != "liquidity":
        symbols = load_index_members(universe_name)
        console.print(f"Using static universe '{universe_name}' ({len(symbols)} symbols)")
    else:
        symbols = broker.list_symbols().symbols
    if args.benchmark_symbol:
        if args.benchmark_symbol not in symbols:
            symbols = list(symbols) + [args.benchmark_symbol]
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    console.print(f"Fetching EOD OHLCV for {len(symbols)} symbols from {start} to {end}...")
    ohlcv = broker.fetch_ohlcv(symbols=symbols, start=start, end=end)
 
    qa = qa_ohlcv_frame(ohlcv)
    if not qa.ok:
        console.print("[red]QA failed[/red]")
        for issue in qa.issues:
            console.print(f"- {issue}")
        raise SystemExit(2)

    # Placeholder: corporate action factors not wired yet (pass None)
    ohlcv_adj = apply_price_adjustments(ohlcv, factors=None)

    out = paths.ohlcv / f"ohlcv_{start.date()}_{end.date()}.parquet"
    write_df_parquet(out, ohlcv_adj)
    console.print(f"Wrote {out}")

    # Universe snapshot (liquidity rule) for end date
    universe_ohlcv = ohlcv_adj
    if args.benchmark_symbol:
        universe_ohlcv = universe_ohlcv[universe_ohlcv["symbol"] != args.benchmark_symbol]
    snap = universe_by_liquidity_rule(
        ohlcv=universe_ohlcv,
        asof=end,
        lookback_days=cfg.universe.lookback_days,
        preselect_top_n=cfg.universe.preselect_top_n,
        final_top_n=cfg.universe.final_top_n,
        min_price=cfg.universe.min_price,
        min_traded_value_inr=cfg.universe.min_traded_value_inr,
    )
    uout = paths.universe / f"universe_{pd.Timestamp(snap.asof).date()}.json"
    write_json(uout, {"asof": str(snap.asof), "symbols": snap.symbols})
    console.print(f"Wrote {uout} (n={len(snap.symbols)})")
 
 
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stock-knight")
    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest EOD OHLCV and build universe snapshot")
    ingest.add_argument("--broker", default="mock", help="Broker provider name (mock for now)")
    ingest.add_argument("--data-root", default="data", help="Root directory for data storage")
    ingest.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ingest.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ingest.add_argument(
        "--creds-file",
        default=None,
        help="Path to YAML credentials file (recommended). Example template: credentials.example.yaml",
    )
    ingest.add_argument(
        "--symbols-file",
        default=None,
        help="Optional file with one symbol per line (recommended for real broker/Yahoo).",
    )
    ingest.add_argument(
        "--universe",
        default="liquidity",
        choices=["liquidity", "nifty50", "nifty100", "nifty150", "nifty500"],
        help=(
            "Symbol selection. 'liquidity' (default) keeps existing behavior "
            "(broker.list_symbols + liquidity rule). "
            "nifty50/100/150/500 read static lists from data/universe_lists/."
        ),
    )
    ingest.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark symbol to also ingest (e.g., ^NSEI for Yahoo, NIFTY 50 for Zerodha).",
    )
    ingest.set_defaults(func=cmd_ingest)

    fl = sub.add_parser("features-labels", help="Build features and forward-return labels")
    fl.add_argument("--data-root", default="data", help="Root directory for data storage")
    fl.add_argument("--ohlcv-parquet", required=True, help="Path to OHLCV parquet")
    fl.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Optional benchmark symbol present in OHLCV to build excess-return labels.",
    )
    fl.set_defaults(
        func=lambda a: build_features_and_labels(
            data_root=Path(a.data_root),
            ohlcv_parquet=Path(a.ohlcv_parquet),
            benchmark_symbol=str(a.benchmark_symbol) if a.benchmark_symbol else None,
        )
    )

    tr = sub.add_parser("train", help="Train models with walk-forward evaluation")
    tr.add_argument("--data-root", default="data", help="Root directory for data storage")
    tr.add_argument("--features-parquet", required=True, help="Path to features parquet")
    tr.add_argument("--labels-parquet", required=True, help="Path to labels parquet")
    tr.add_argument(
        "--model-kind", default="ridge", choices=["ridge", "lgbm", "xgb", "rf"]
    )
    tr.add_argument(
        "--target",
        default="raw",
        choices=["raw", "excess"],
        help="Train on raw forward returns or benchmark-relative excess returns (requires *_excess labels).",
    )
    tr.set_defaults(
        func=lambda a: train_walk_forward(
            data_root=Path(a.data_root),
            features_parquet=Path(a.features_parquet),
            labels_parquet=Path(a.labels_parquet),
            model_kind=str(a.model_kind),
            target=str(a.target),
        )
    )

    bt = sub.add_parser("backtest", help="Run a simple cost-aware backtest (MVP)")
    bt.add_argument("--features-parquet", required=True)
    bt.add_argument("--labels-parquet", required=True)
    bt.add_argument("--model-path", required=True)
    bt.add_argument("--horizon", default="1d", choices=["1d", "7d", "1m", "6m", "12m"])
    bt.add_argument("--k", type=int, default=20, help="Top-k for long book (and short book if enabled)")
    bt.add_argument("--long-short", action="store_true", help="Enable long/short (research mode)")
    bt.set_defaults(
        func=lambda a: run_backtest(
            features_parquet=Path(a.features_parquet),
            labels_parquet=Path(a.labels_parquet),
            model_path=Path(a.model_path),
            horizon=str(a.horizon),
            spec=PortfolioSpec(long_k=int(a.k), short_k=int(a.k), long_short=bool(a.long_short)),
        )
    )

    rk = sub.add_parser("rank", help="Generate rank lists for a date")
    rk.add_argument("--features-parquet", required=True)
    rk.add_argument("--ohlcv-parquet", default=None, help="Optional OHLCV parquet to enrich with prices")
    rk.add_argument("--model-path", required=True)
    rk.add_argument("--asof", required=True, help="Date YYYY-MM-DD")
    rk.add_argument("--horizon", required=True, choices=["1d", "7d", "1m", "6m", "12m"])
    rk.add_argument(
        "--training-target",
        default="raw",
        choices=["raw", "excess"],
        help="Metadata only: records which training target you used when producing this rank file.",
    )
    rk.add_argument(
        "--benchmark-symbol",
        default=None,
        help="Metadata only: records benchmark symbol (when training-target=excess).",
    )
    rk.add_argument("--out", default="reports/ranks.json", help="Output JSON path")
    rk.add_argument(
        "--live-ltp",
        action="store_true",
        help="Fetch broker last traded price for ranked symbols (requires --ltp-broker).",
    )
    rk.add_argument(
        "--ltp-broker",
        default=None,
        choices=["yahoo", "yahoofinance", "zerodha", "kite", "angelone", "angel", "upstox"],
        help="Broker for live LTP. Zerodha/Angel One/Upstox need --creds-file.",
    )
    rk.add_argument(
        "--creds-file",
        default=None,
        help="YAML credentials path (needed for Zerodha live LTP).",
    )
    rk.add_argument(
        "--universe",
        default="liquidity",
        choices=["liquidity", "nifty50", "nifty100", "nifty150", "nifty500"],
        help=(
            "Restrict ranked output to a NIFTY index membership list. "
            "'liquidity' (default) ranks all symbols present in the features parquet."
        ),
    )
    rk.add_argument(
        "--rank-method",
        default="composite",
        choices=["composite", "score_only"],
        help=(
            "How to sort ranked output. 'composite' (default) blends predicted "
            "return, Sharpe, momentum, low-vol, and model confidence. "
            "'score_only' preserves the legacy single-score sort."
        ),
    )
    rk.set_defaults(func=cmd_rank)

    ex = sub.add_parser(
        "execute",
        help="Zerodha order execution: funds | holdings | orders | plan | execute",
    )
    ex.add_argument(
        "action",
        choices=["funds", "holdings", "orders", "plan", "execute"],
        help=(
            "funds     – show available cash\n"
            "holdings  – show current Demat holdings\n"
            "orders    – show today's orders\n"
            "plan      – build & preview order plan (dry-run)\n"
            "execute   – build plan and place orders (use --live to go live)"
        ),
    )
    ex.add_argument("--broker", default="zerodha", choices=["zerodha", "kite", "angelone", "angel"])
    ex.add_argument("--creds-file", default="credentials.yaml", help="YAML credentials path")
    ex.add_argument("--rank-file", default=None, help="Path to rank JSON (required for plan/execute)")
    ex.add_argument("--top-k", type=int, default=10, help="Target top-K stocks")
    ex.add_argument(
        "--capital",
        type=float,
        default=None,
        help="INR to allocate (default: full available cash)",
    )
    ex.add_argument(
        "--no-sell-exits",
        action="store_true",
        help="Skip SELL orders for holdings no longer in top-K",
    )
    ex.add_argument(
        "--order-type",
        default="MARKET",
        choices=["MARKET", "LIMIT"],
        help="Order type (LIMIT uses LTP as price)",
    )
    ex.add_argument(
        "--product",
        default="CNC",
        choices=["CNC", "MIS"],
        help="CNC=delivery (overnight), MIS=intraday",
    )
    ex.add_argument(
        "--live",
        action="store_true",
        help="Actually place orders on Zerodha (default is dry-run)",
    )
    ex.add_argument("--out", default=None, help="Save execution plan/results to this JSON file")
    ex.set_defaults(func=cmd_execute)

    dr = sub.add_parser("drift", help="Compute simple IC drift report from stored features+labels")
    dr.add_argument("--features-parquet", required=True)
    dr.add_argument("--labels-parquet", required=True)
    dr.add_argument("--model-path", required=True)
    dr.add_argument("--horizon", required=True, choices=["1d", "7d", "1m", "6m", "12m"])
    dr.add_argument("--window-days", type=int, default=252)
    dr.add_argument("--out", default="reports/drift.json")
    dr.set_defaults(
        func=lambda a: _cmd_drift(
            features_parquet=Path(a.features_parquet),
            labels_parquet=Path(a.labels_parquet),
            model_path=Path(a.model_path),
            horizon=str(a.horizon),
            window_days=int(a.window_days),
            out=Path(a.out),
        )
    )

    return p


def _cmd_drift(
    features_parquet: Path,
    labels_parquet: Path,
    model_path: Path,
    horizon: str,
    window_days: int,
    out: Path,
) -> None:
    feats = read_df_parquet(features_parquet)
    labels = read_df_parquet(labels_parquet)
    tm = load_model(model_path)

    df = feats.join(labels, how="inner").dropna(subset=tm.feature_cols)

    horizon_to_col = {
        "1d": "r_fwd_1d",
        "7d": "r_fwd_7d",
        "1m": "r_fwd_1m",
        "6m": "r_fwd_6m",
        "12m": "r_fwd_12m",
    }
    ycol = horizon_to_col[horizon]
    df = df.dropna(subset=[ycol])

    scores = pd.Series(tm.predict(df), index=df.index, name="score")
    report = drift_report_from_scores(scores=scores, labels=df[ycol], horizon=horizon, window_days=window_days)
    write_drift_report(out, report)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
 
