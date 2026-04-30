from __future__ import annotations
 
import argparse
from pathlib import Path
 
import pandas as pd
from rich.console import Console
 
from stock_knight.config import ProjectConfig, default_config
from stock_knight.data.corporate_actions import apply_price_adjustments
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
        return UpstoxEODClient(UpstoxConfig(access_token=access_token))
    if name in {"angelone", "angel"}:
        import os

        from stock_knight.brokers.angelone import AngelOneConfig, AngelOneSmartApiEODClient

        a = creds.get("angelone", {}) if isinstance(creds, dict) else {}
        api_key = a.get("api_key") or os.environ.get("ANGELONE_API_KEY")
        client_code = a.get("client_code") or os.environ.get("ANGELONE_CLIENT_CODE")
        password = a.get("password") or os.environ.get("ANGELONE_PASSWORD")
        totp = a.get("totp") or os.environ.get("ANGELONE_TOTP")
        if not (api_key and client_code and password and totp):
            raise ValueError(
                "Missing Angel One credentials. Provide --creds-file with angelone.* "
                "or set env vars: ANGELONE_API_KEY, ANGELONE_CLIENT_CODE, ANGELONE_PASSWORD, ANGELONE_TOTP."
            )
        return AngelOneSmartApiEODClient(
            AngelOneConfig(api_key=api_key, client_code=client_code, password=password, totp=totp)
        )
    raise ValueError(f"Unknown broker={name}. Use one of: mock, yahoo, zerodha, upstox, angelone.")
 
 
def cmd_ingest(args: argparse.Namespace) -> None:
    console = Console()
    cfg: ProjectConfig = default_config()
    root = Path(args.data_root)
    paths = DataPaths(root=root)
    ensure_dirs(paths.ohlcv, paths.corporate_actions, paths.universe)

    creds = load_credentials(args.creds_file)
    broker = _get_broker_client(args.broker, creds=creds)
    if args.symbols_file:
        symbols = [s.strip() for s in Path(args.symbols_file).read_text().splitlines() if s.strip()]
    else:
        symbols = broker.list_symbols().symbols
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
    snap = universe_by_liquidity_rule(
        ohlcv=ohlcv_adj,
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
    ingest.set_defaults(func=cmd_ingest)

    fl = sub.add_parser("features-labels", help="Build features and forward-return labels")
    fl.add_argument("--data-root", default="data", help="Root directory for data storage")
    fl.add_argument("--ohlcv-parquet", required=True, help="Path to OHLCV parquet")
    fl.set_defaults(
        func=lambda a: build_features_and_labels(
            data_root=Path(a.data_root),
            ohlcv_parquet=Path(a.ohlcv_parquet),
        )
    )

    tr = sub.add_parser("train", help="Train models with walk-forward evaluation")
    tr.add_argument("--data-root", default="data", help="Root directory for data storage")
    tr.add_argument("--features-parquet", required=True, help="Path to features parquet")
    tr.add_argument("--labels-parquet", required=True, help="Path to labels parquet")
    tr.add_argument("--model-kind", default="ridge", choices=["ridge", "lgbm"])
    tr.set_defaults(
        func=lambda a: train_walk_forward(
            data_root=Path(a.data_root),
            features_parquet=Path(a.features_parquet),
            labels_parquet=Path(a.labels_parquet),
            model_kind=str(a.model_kind),
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

    rk = sub.add_parser("rank", help="Generate Top10/20/30 lists for a date")
    rk.add_argument("--features-parquet", required=True)
    rk.add_argument("--model-path", required=True)
    rk.add_argument("--asof", required=True, help="Date YYYY-MM-DD")
    rk.add_argument("--horizon", required=True, choices=["1d", "7d", "1m", "6m", "12m"])
    rk.add_argument("--out", default="reports/ranks.json", help="Output JSON path")
    rk.set_defaults(
        func=lambda a: write_ranklists(
            Path(a.out),
            rank_for_date(
                features_parquet=Path(a.features_parquet),
                model_path=Path(a.model_path),
                asof=pd.Timestamp(a.asof),
                horizon=str(a.horizon),
            ),
        )
    )

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
 
