#!/usr/bin/env python3
"""
Binance USD-M Futures 09:00 KST daily rank scanner

What it does
------------
For every Binance USD-M USDT perpetual symbol, it calculates the 24h return
for each Binance 1d candle. Binance USD-M 1d candles open at 00:00 UTC,
which is 09:00 in Asia/Seoul. Therefore, each completed 1d candle represents:

    KST 09:00 -> next day KST 09:00

At each KST 09:00 snapshot, the script ranks all symbols by the just-completed
24h return and saves:

    1) all_returns_by_9am.csv
    2) top_bottom_by_9am.csv
    3) latest_9am_top_bottom.csv
    4) market_breadth_by_9am.csv
    5) rank_forward_performance_summary.csv
    6) side_forward_performance_summary.csv

The forward-performance columns answer questions like:

    "After a coin was top 10 at 09:00, did it continue rising over the next 24h?"
    "After a coin was bottom 10 at 09:00, did it rebound or keep falling?"

This is for research only. It is not investment advice and does not place orders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_URL = "https://fapi.binance.com"
KST_TZ = "Asia/Seoul"


@dataclass
class Config:
    days: int
    top_n: int
    outdir: str
    min_quote_volume: float
    sleep_sec: float
    timeout: int
    max_symbols: Optional[int]
    symbols: Optional[List[str]]
    exclude_symbols: List[str]
    save_excel: bool


def request_json(path: str, params: Optional[dict] = None, *, timeout: int = 20, retries: int = 3, sleep_sec: float = 0.25) -> Any:
    url = BASE_URL + path
    params = params or {}
    last_exc: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code} for {url}\n"
                    f"params={params}\n"
                    f"response={response.text[:1000]}"
                )
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries:
                time.sleep(sleep_sec * attempt)

    raise RuntimeError(f"Request failed for {url} params={params}: {last_exc}")


def utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def now_ms_utc() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def get_usdt_perp_symbols(timeout: int, sleep_sec: float) -> List[str]:
    data = request_json("/fapi/v1/exchangeInfo", timeout=timeout, sleep_sec=sleep_sec)
    symbols: List[str] = []
    for item in data.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("contractType") == "PERPETUAL"
        ):
            symbols.append(item["symbol"])
    return sorted(symbols)


def normalize_symbol_list(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    out = []
    for x in raw.split(","):
        x = x.strip().upper()
        if x:
            out.append(x)
    return out or None


def fetch_daily_klines(symbol: str, start_ms: int, end_ms: int, timeout: int, sleep_sec: float) -> pd.DataFrame:
    rows: List[list] = []
    cursor = start_ms

    while cursor < end_ms:
        data = request_json(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1500,
            },
            timeout=timeout,
            sleep_sec=sleep_sec,
        )
        if not data:
            break

        rows.extend(data)
        last_open = int(data[-1][0])
        next_cursor = last_open + 24 * 60 * 60 * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        # Usually one request is enough for this use case, but keep pagination safe.
        if len(data) < 1500:
            break
        time.sleep(sleep_sec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "base_volume",
            "close_time_ms",
            "quote_volume",
            "num_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "base_volume",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
    df["close_time_ms"] = pd.to_numeric(df["close_time_ms"], errors="coerce").astype("int64")
    df["symbol"] = symbol

    # Exclude currently forming daily candle.
    current_ms = now_ms_utc()
    df = df[df["close_time_ms"] < current_ms].copy()

    return df


def build_returns_frame(symbols: List[str], cfg: Config) -> pd.DataFrame:
    # Fetch a little more than requested because snapshot date is candle close date in KST.
    end_ms = now_ms_utc()
    start_ms = end_ms - (cfg.days + 3) * 24 * 60 * 60 * 1000

    frames: List[pd.DataFrame] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        print(f"[{i:>4}/{total}] Fetching 1d klines: {symbol}")
        try:
            df = fetch_daily_klines(symbol, start_ms, end_ms, cfg.timeout, cfg.sleep_sec)
            if df.empty:
                continue
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  - skipped {symbol}: {exc}")
        time.sleep(cfg.sleep_sec)

    if not frames:
        raise RuntimeError("No kline data was downloaded.")

    raw = pd.concat(frames, ignore_index=True)

    open_utc = pd.to_datetime(raw["open_time_ms"], unit="ms", utc=True)
    close_snapshot_utc = open_utc + pd.Timedelta(days=1)
    raw["period_start_kst"] = open_utc.dt.tz_convert(KST_TZ)
    raw["snapshot_kst_9am"] = close_snapshot_utc.dt.tz_convert(KST_TZ)
    raw["snapshot_date_kst"] = raw["snapshot_kst_9am"].dt.strftime("%Y-%m-%d")

    raw["return_24h_pct"] = (raw["close"] / raw["open"] - 1.0) * 100.0
    raw["high_from_open_pct"] = (raw["high"] / raw["open"] - 1.0) * 100.0
    raw["low_from_open_pct"] = (raw["low"] / raw["open"] - 1.0) * 100.0
    raw["range_pct"] = (raw["high"] / raw["low"] - 1.0) * 100.0
    raw["taker_buy_quote_share"] = raw["taker_buy_quote_volume"] / raw["quote_volume"].replace({0: math.nan})

    raw = raw.sort_values(["symbol", "open_time_ms"]).reset_index(drop=True)
    raw["next_24h_return_pct"] = raw.groupby("symbol")["return_24h_pct"].shift(-1)
    raw["next_24h_quote_volume"] = raw.groupby("symbol")["quote_volume"].shift(-1)

    # Keep only requested number of completed 09:00 snapshots.
    unique_dates = sorted(raw["snapshot_date_kst"].dropna().unique())
    keep_dates = set(unique_dates[-cfg.days :])
    raw = raw[raw["snapshot_date_kst"].isin(keep_dates)].copy()

    if cfg.min_quote_volume > 0:
        raw = raw[raw["quote_volume"] >= cfg.min_quote_volume].copy()

    return raw


def build_rank_tables(all_returns: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_returns = all_returns.sort_values(["snapshot_date_kst", "return_24h_pct"], ascending=[True, False]).copy()
    all_returns["rank_up"] = all_returns.groupby("snapshot_date_kst")["return_24h_pct"].rank(method="first", ascending=False).astype(int)
    all_returns["rank_down"] = all_returns.groupby("snapshot_date_kst")["return_24h_pct"].rank(method="first", ascending=True).astype(int)

    up = all_returns[all_returns["rank_up"] <= top_n].copy()
    up["side"] = "UP_TOP"
    up["rank"] = up["rank_up"]

    down = all_returns[all_returns["rank_down"] <= top_n].copy()
    down["side"] = "DOWN_TOP"
    down["rank"] = down["rank_down"]

    top_bottom = pd.concat([up, down], ignore_index=True)
    top_bottom = top_bottom.sort_values(["snapshot_date_kst", "side", "rank"]).reset_index(drop=True)

    cols = [
        "snapshot_date_kst",
        "snapshot_kst_9am",
        "side",
        "rank",
        "symbol",
        "return_24h_pct",
        "next_24h_return_pct",
        "open",
        "high",
        "low",
        "close",
        "quote_volume",
        "num_trades",
        "taker_buy_quote_share",
        "high_from_open_pct",
        "low_from_open_pct",
        "range_pct",
    ]
    top_bottom = top_bottom[cols]

    latest_date = top_bottom["snapshot_date_kst"].max()
    latest = top_bottom[top_bottom["snapshot_date_kst"] == latest_date].copy()

    breadth = (
        all_returns.groupby("snapshot_date_kst")
        .agg(
            snapshot_kst_9am=("snapshot_kst_9am", "max"),
            symbols_count=("symbol", "count"),
            average_return_24h_pct=("return_24h_pct", "mean"),
            median_return_24h_pct=("return_24h_pct", "median"),
            up_count=("return_24h_pct", lambda s: int((s > 0).sum())),
            down_count=("return_24h_pct", lambda s: int((s < 0).sum())),
            average_quote_volume=("quote_volume", "mean"),
            total_quote_volume=("quote_volume", "sum"),
        )
        .reset_index()
    )
    breadth["up_ratio"] = breadth["up_count"] / breadth["symbols_count"].replace({0: math.nan})

    # Forward performance: what happened in the next 24h after a rank signal at 09:00?
    perf_base = top_bottom.dropna(subset=["next_24h_return_pct"]).copy()
    rank_perf = (
        perf_base.groupby(["side", "rank"])
        .agg(
            count=("next_24h_return_pct", "count"),
            mean_next_24h_return_pct=("next_24h_return_pct", "mean"),
            median_next_24h_return_pct=("next_24h_return_pct", "median"),
            win_rate_next_24h_up=("next_24h_return_pct", lambda s: float((s > 0).mean())),
            avg_signal_return_24h_pct=("return_24h_pct", "mean"),
        )
        .reset_index()
        .sort_values(["side", "rank"])
    )

    side_perf = (
        perf_base.groupby("side")
        .agg(
            count=("next_24h_return_pct", "count"),
            mean_next_24h_return_pct=("next_24h_return_pct", "mean"),
            median_next_24h_return_pct=("next_24h_return_pct", "median"),
            win_rate_next_24h_up=("next_24h_return_pct", lambda s: float((s > 0).mean())),
            avg_signal_return_24h_pct=("return_24h_pct", "mean"),
            avg_quote_volume=("quote_volume", "mean"),
        )
        .reset_index()
        .sort_values("side")
    )

    return all_returns, top_bottom, latest, breadth, rank_perf, side_perf


def save_outputs(
    cfg: Config,
    all_returns: pd.DataFrame,
    top_bottom: pd.DataFrame,
    latest: pd.DataFrame,
    breadth: pd.DataFrame,
    rank_perf: pd.DataFrame,
    side_perf: pd.DataFrame,
    symbols: List[str],
) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)

    all_returns.to_csv(os.path.join(cfg.outdir, "all_returns_by_9am.csv"), index=False, encoding="utf-8-sig")
    top_bottom.to_csv(os.path.join(cfg.outdir, "top_bottom_by_9am.csv"), index=False, encoding="utf-8-sig")
    latest.to_csv(os.path.join(cfg.outdir, "latest_9am_top_bottom.csv"), index=False, encoding="utf-8-sig")
    breadth.to_csv(os.path.join(cfg.outdir, "market_breadth_by_9am.csv"), index=False, encoding="utf-8-sig")
    rank_perf.to_csv(os.path.join(cfg.outdir, "rank_forward_performance_summary.csv"), index=False, encoding="utf-8-sig")
    side_perf.to_csv(os.path.join(cfg.outdir, "side_forward_performance_summary.csv"), index=False, encoding="utf-8-sig")

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": KST_TZ,
        "definition": "Each snapshot is KST 09:00. return_24h_pct is the completed Binance USD-M 1d candle return from previous KST 09:00 to snapshot KST 09:00.",
        "days": cfg.days,
        "top_n": cfg.top_n,
        "min_quote_volume": cfg.min_quote_volume,
        "symbols_count_requested": len(symbols),
        "symbols": symbols,
        "outputs": [
            "all_returns_by_9am.csv",
            "top_bottom_by_9am.csv",
            "latest_9am_top_bottom.csv",
            "market_breadth_by_9am.csv",
            "rank_forward_performance_summary.csv",
            "side_forward_performance_summary.csv",
        ],
    }
    with open(os.path.join(cfg.outdir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if cfg.save_excel:
        xlsx_path = os.path.join(cfg.outdir, "binance_9am_rank_analysis.xlsx")
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                latest.to_excel(writer, sheet_name="latest_top_bottom", index=False)
                top_bottom.to_excel(writer, sheet_name="top_bottom_by_9am", index=False)
                breadth.to_excel(writer, sheet_name="market_breadth", index=False)
                rank_perf.to_excel(writer, sheet_name="rank_forward_perf", index=False)
                side_perf.to_excel(writer, sheet_name="side_forward_perf", index=False)
                all_returns.to_excel(writer, sheet_name="all_returns", index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Excel save failed. CSV files are still saved. Reason: {exc}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Binance USD-M Futures 09:00 KST top/bottom daily return scanner")
    parser.add_argument("--days", type=int, default=30, help="Number of completed KST 09:00 snapshots to analyze")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top gainers and losers per day")
    parser.add_argument("--outdir", default="outputs_binance_9am_rank", help="Output directory")
    parser.add_argument("--min-quote-volume", type=float, default=0.0, help="Minimum 24h quote volume in USDT. Example: 10000000")
    parser.add_argument("--sleep-sec", type=float, default=0.08, help="Sleep seconds between API calls")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    parser.add_argument("--max-symbols", type=int, default=None, help="Debug option: limit number of symbols")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols. If omitted, use all trading USDT perpetuals")
    parser.add_argument("--exclude-symbols", default="", help="Comma-separated symbols to exclude")
    parser.add_argument("--save-excel", action="store_true", help="Also save an Excel workbook. Requires openpyxl")
    args = parser.parse_args()

    if args.days < 2:
        raise ValueError("--days must be at least 2")
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")

    return Config(
        days=args.days,
        top_n=args.top_n,
        outdir=args.outdir,
        min_quote_volume=args.min_quote_volume,
        sleep_sec=args.sleep_sec,
        timeout=args.timeout,
        max_symbols=args.max_symbols,
        symbols=normalize_symbol_list(args.symbols),
        exclude_symbols=normalize_symbol_list(args.exclude_symbols) or [],
        save_excel=args.save_excel,
    )


def main() -> None:
    cfg = parse_args()

    if cfg.symbols:
        symbols = sorted(set(cfg.symbols))
    else:
        print("Fetching Binance USD-M Futures exchangeInfo...")
        symbols = get_usdt_perp_symbols(cfg.timeout, cfg.sleep_sec)

    if cfg.exclude_symbols:
        exclude = set(cfg.exclude_symbols)
        symbols = [s for s in symbols if s not in exclude]

    if cfg.max_symbols is not None:
        symbols = symbols[: cfg.max_symbols]

    print(f"Symbols selected: {len(symbols)}")
    print(f"Output directory: {cfg.outdir}")

    all_returns = build_returns_frame(symbols, cfg)
    all_returns, top_bottom, latest, breadth, rank_perf, side_perf = build_rank_tables(all_returns, cfg.top_n)
    save_outputs(cfg, all_returns, top_bottom, latest, breadth, rank_perf, side_perf, symbols)

    print("\nDone.")
    print(f"Saved: {cfg.outdir}/all_returns_by_9am.csv")
    print(f"Saved: {cfg.outdir}/top_bottom_by_9am.csv")
    print(f"Saved: {cfg.outdir}/latest_9am_top_bottom.csv")
    print(f"Saved: {cfg.outdir}/market_breadth_by_9am.csv")
    print(f"Saved: {cfg.outdir}/rank_forward_performance_summary.csv")
    print(f"Saved: {cfg.outdir}/side_forward_performance_summary.csv")

    if not latest.empty:
        print("\nLatest KST 09:00 ranking:")
        show_cols = ["snapshot_date_kst", "side", "rank", "symbol", "return_24h_pct", "next_24h_return_pct", "quote_volume"]
        print(latest[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
