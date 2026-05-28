"""Side-specific audit for Binance 9AM rank model outputs.

Reads one or more model output directories produced by train_binance_9am_rank_model.py
and audits LONG-only, SHORT-only, and BOTH daily strategies with transaction cost.

Examples, PowerShell:
  python audit_9am_rank_sides.py --dirs outputs_9am_rank_model_365_vol100m_test180 --top-n 5 --cost-pct 0.20
  python audit_9am_rank_sides.py --dirs outputs_9am_rank_model_365_vol100m_test180 --top-n 3 --cost-pct 0.50 --long-thresholds 0.55 0.60 --short-thresholds 0.45 0.40
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_predictions(model_dir: Path) -> pd.DataFrame:
    candidates = [model_dir / "test_predictions.csv", model_dir / "latest_predictions.csv"]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"No test_predictions.csv or latest_predictions.csv found in {model_dir}")

    df = pd.read_csv(path)
    needed = {"snapshot_date_kst", "symbol", "prob_up"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    if "target_trade_return_long_pct" not in df.columns:
        if "next_24h_return_pct" not in df.columns:
            raise ValueError(f"{path} needs target_trade_return_long_pct or next_24h_return_pct")
        df["target_trade_return_long_pct"] = _as_float(df["next_24h_return_pct"])

    if "target_trade_return_short_pct" not in df.columns:
        df["target_trade_return_short_pct"] = -_as_float(df["target_trade_return_long_pct"])

    df["prob_up"] = _as_float(df["prob_up"])
    df["target_trade_return_long_pct"] = _as_float(df["target_trade_return_long_pct"])
    df["target_trade_return_short_pct"] = _as_float(df["target_trade_return_short_pct"])
    df = df.dropna(subset=["snapshot_date_kst", "symbol", "prob_up"])
    return df


def summarize_daily(daily: pd.DataFrame, ret_col: str) -> Dict[str, Any]:
    valid = daily.dropna(subset=[ret_col]).copy()
    if valid.empty:
        return {
            "days": 0,
            "avg_daily_pct_net": None,
            "median_daily_pct_net": None,
            "daily_win_rate": None,
            "worst_day_pct_net": None,
            "best_day_pct_net": None,
            "top5_days_share_of_sum": None,
            "gross_compound_multiple_net": None,
        }

    r = valid[ret_col].astype(float)
    total = float(r.sum())
    top5 = float(r.nlargest(min(5, len(r))).sum())
    top5_share = None if abs(total) < 1e-12 else top5 / total
    gross_mult = float(np.prod(1.0 + r / 100.0))
    return {
        "days": int(len(valid)),
        "avg_daily_pct_net": float(r.mean()),
        "median_daily_pct_net": float(r.median()),
        "daily_win_rate": float((r > 0).mean()),
        "worst_day_pct_net": float(r.min()),
        "best_day_pct_net": float(r.max()),
        "top5_days_share_of_sum": top5_share,
        "gross_compound_multiple_net": gross_mult,
    }


def audit_one(
    model_dir: Path,
    top_n: int,
    cost_pct: float,
    long_thresholds: Iterable[Optional[float]],
    short_thresholds: Iterable[Optional[float]],
) -> Dict[str, Any]:
    df = load_predictions(model_dir)
    # Exclude the latest rows without next-24h target.
    hist = df.dropna(subset=["target_trade_return_long_pct", "target_trade_return_short_pct"]).copy()

    summary: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "rows_with_targets": int(len(hist)),
        "date_start": None if hist.empty else str(hist["snapshot_date_kst"].min()),
        "date_end": None if hist.empty else str(hist["snapshot_date_kst"].max()),
        "top_n": int(top_n),
        "cost_pct": float(cost_pct),
        "audits": [],
    }

    if hist.empty:
        return summary

    # LONG-only audits.
    for thr in long_thresholds:
        parts = []
        for date, g in hist.groupby("snapshot_date_kst", sort=True):
            gg = g if thr is None else g[g["prob_up"] >= thr]
            pick = gg.sort_values("prob_up", ascending=False).head(top_n)
            ret = np.nan if pick.empty else float(pick["target_trade_return_long_pct"].mean() - cost_pct)
            parts.append({"snapshot_date_kst": date, "side": "LONG", "threshold": thr, "daily_return_pct_net": ret, "num_positions": int(len(pick))})
        daily = pd.DataFrame(parts)
        item = {"side": "LONG", "threshold": thr, **summarize_daily(daily, "daily_return_pct_net")}
        out = model_dir / f"audit_long_top{top_n}_cost{cost_pct}_thr{thr if thr is not None else 'none'}.csv"
        daily.to_csv(out, index=False, encoding="utf-8-sig")
        item["daily_file"] = str(out)
        summary["audits"].append(item)

    # SHORT-only audits.
    for thr in short_thresholds:
        parts = []
        for date, g in hist.groupby("snapshot_date_kst", sort=True):
            gg = g if thr is None else g[g["prob_up"] <= thr]
            pick = gg.sort_values("prob_up", ascending=True).head(top_n)
            ret = np.nan if pick.empty else float(pick["target_trade_return_short_pct"].mean() - cost_pct)
            parts.append({"snapshot_date_kst": date, "side": "SHORT", "threshold": thr, "daily_return_pct_net": ret, "num_positions": int(len(pick))})
        daily = pd.DataFrame(parts)
        item = {"side": "SHORT", "threshold": thr, **summarize_daily(daily, "daily_return_pct_net")}
        out = model_dir / f"audit_short_top{top_n}_cost{cost_pct}_thr{thr if thr is not None else 'none'}.csv"
        daily.to_csv(out, index=False, encoding="utf-8-sig")
        item["daily_file"] = str(out)
        summary["audits"].append(item)

    # BOTH side: average of available long and short daily legs.
    for lthr in long_thresholds:
        for sthr in short_thresholds:
            parts = []
            for date, g in hist.groupby("snapshot_date_kst", sort=True):
                lg = g if lthr is None else g[g["prob_up"] >= lthr]
                sg = g if sthr is None else g[g["prob_up"] <= sthr]
                long_pick = lg.sort_values("prob_up", ascending=False).head(top_n)
                short_pick = sg.sort_values("prob_up", ascending=True).head(top_n)
                vals = []
                if not long_pick.empty:
                    vals.append(float(long_pick["target_trade_return_long_pct"].mean() - cost_pct))
                if not short_pick.empty:
                    vals.append(float(short_pick["target_trade_return_short_pct"].mean() - cost_pct))
                ret = np.nan if not vals else float(np.mean(vals))
                parts.append({
                    "snapshot_date_kst": date,
                    "side": "BOTH",
                    "long_threshold": lthr,
                    "short_threshold": sthr,
                    "daily_return_pct_net": ret,
                    "num_long": int(len(long_pick)),
                    "num_short": int(len(short_pick)),
                })
            daily = pd.DataFrame(parts)
            item = {
                "side": "BOTH",
                "long_threshold": lthr,
                "short_threshold": sthr,
                **summarize_daily(daily, "daily_return_pct_net"),
            }
            out = model_dir / f"audit_both_top{top_n}_cost{cost_pct}_long{lthr if lthr is not None else 'none'}_short{sthr if sthr is not None else 'none'}.csv"
            daily.to_csv(out, index=False, encoding="utf-8-sig")
            item["daily_file"] = str(out)
            summary["audits"].append(item)

    out_json = model_dir / f"audit_sides_top{top_n}_cost{cost_pct}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_file"] = str(out_json)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="Model output directories")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--cost-pct", type=float, default=0.20, help="Round-trip cost per selected leg in percent")
    ap.add_argument("--long-thresholds", nargs="*", type=float, default=[0.55, 0.60, 0.65])
    ap.add_argument("--short-thresholds", nargs="*", type=float, default=[0.45, 0.40, 0.35])
    ap.add_argument("--include-no-threshold", action="store_true")
    args = ap.parse_args()

    long_thrs: List[Optional[float]] = list(args.long_thresholds)
    short_thrs: List[Optional[float]] = list(args.short_thresholds)
    if args.include_no_threshold:
        long_thrs = [None] + long_thrs
        short_thrs = [None] + short_thrs

    all_summaries = []
    for d in args.dirs:
        model_dir = Path(d)
        print(f"Auditing {model_dir} ...")
        summary = audit_one(model_dir, args.top_n, args.cost_pct, long_thrs, short_thrs)
        all_summaries.append(summary)
        print(f"  saved: {summary.get('summary_file')}")
        for item in summary["audits"]:
            side = item.get("side")
            if side == "BOTH":
                label = f"BOTH L>={item.get('long_threshold')} S<={item.get('short_threshold')}"
            else:
                label = f"{side} thr={item.get('threshold')}"
            print(
                f"  {label}: days={item.get('days')} avg={item.get('avg_daily_pct_net')} "
                f"median={item.get('median_daily_pct_net')} win={item.get('daily_win_rate')} "
                f"worst={item.get('worst_day_pct_net')} gross={item.get('gross_compound_multiple_net')}"
            )

    Path("audit_9am_rank_sides_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved combined summary: audit_9am_rank_sides_summary.json")


if __name__ == "__main__":
    main()
