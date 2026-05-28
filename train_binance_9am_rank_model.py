#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an AI model from Binance KST 09:00 ranking scanner outputs.

Input files expected from binance_9am_rank_scanner.py:
  - all_returns_by_9am.csv
  - top_bottom_by_9am.csv          optional, for sample-mode=top_bottom
  - market_breadth_by_9am.csv      optional, joined if available

Main target:
  next_24h_return_pct > 0

This script is research/backtest tooling only. It does not place orders.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


RANDOM_STATE = 42


@dataclass
class Config:
    data_dir: str
    all_returns: Optional[str]
    top_bottom: Optional[str]
    market_breadth: Optional[str]
    sample_mode: str
    outdir: str
    min_quote_volume: float
    test_days: int
    top_k: int
    probability_threshold: float
    model: str
    target_return_threshold: float
    drop_symbols_containing: str


def make_onehot_encoder() -> OneHotEncoder:
    """scikit-learn changed sparse -> sparse_output. Support both."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_input_files(cfg: Config) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    data_dir = Path(cfg.data_dir)
    all_path = Path(cfg.all_returns) if cfg.all_returns else data_dir / "all_returns_by_9am.csv"
    top_path = Path(cfg.top_bottom) if cfg.top_bottom else data_dir / "top_bottom_by_9am.csv"
    breadth_path = Path(cfg.market_breadth) if cfg.market_breadth else data_dir / "market_breadth_by_9am.csv"

    if cfg.sample_mode == "top_bottom":
        if not top_path.exists():
            raise FileNotFoundError(f"top_bottom file not found: {top_path}")
        df = pd.read_csv(top_path)
    else:
        if not all_path.exists():
            raise FileNotFoundError(f"all_returns file not found: {all_path}")
        df = pd.read_csv(all_path)

    breadth = pd.read_csv(breadth_path) if breadth_path.exists() else None
    return df, breadth


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_col = find_col(df, ["snapshot_date_kst", "date", "snapshot_date", "day", "open_date"])
    if date_col is None:
        raise ValueError("Could not find date column. Expected snapshot_date_kst or similar.")
    if date_col != "snapshot_date_kst":
        df = df.rename(columns={date_col: "snapshot_date_kst"})

    symbol_col = find_col(df, ["symbol", "ticker"])
    if symbol_col is None:
        raise ValueError("Could not find symbol column.")
    if symbol_col != "symbol":
        df = df.rename(columns={symbol_col: "symbol"})

    ret_col = find_col(df, ["return_24h_pct", "ret_24h_pct", "return_pct", "ret_pct"])
    if ret_col is None:
        raise ValueError("Could not find current 24h return column. Expected return_24h_pct.")
    if ret_col != "return_24h_pct":
        df = df.rename(columns={ret_col: "return_24h_pct"})

    next_col = find_col(df, ["next_24h_return_pct", "forward_24h_return_pct", "target_return_pct"])
    if next_col is not None and next_col != "next_24h_return_pct":
        df = df.rename(columns={next_col: "next_24h_return_pct"})
    elif next_col is None:
        df["next_24h_return_pct"] = np.nan

    vol_col = find_col(df, ["quote_volume", "quote_asset_volume", "volume_usdt", "turnover"])
    if vol_col is not None and vol_col != "quote_volume":
        df = df.rename(columns={vol_col: "quote_volume"})
    elif vol_col is None:
        df["quote_volume"] = np.nan

    if "side" not in df.columns:
        df["side"] = "ALL"
    if "rank" not in df.columns:
        df["rank"] = np.nan

    df["snapshot_date_kst"] = pd.to_datetime(df["snapshot_date_kst"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str)
    df["return_24h_pct"] = safe_float_series(df["return_24h_pct"])
    df["next_24h_return_pct"] = safe_float_series(df["next_24h_return_pct"])
    df["quote_volume"] = safe_float_series(df["quote_volume"])
    df["rank"] = safe_float_series(df["rank"])

    return df


def normalize_breadth(breadth: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if breadth is None or breadth.empty:
        return None
    b = breadth.copy()
    date_col = find_col(b, ["snapshot_date_kst", "date", "snapshot_date", "day"])
    if date_col is None:
        return None
    if date_col != "snapshot_date_kst":
        b = b.rename(columns={date_col: "snapshot_date_kst"})
    b["snapshot_date_kst"] = pd.to_datetime(b["snapshot_date_kst"], errors="coerce")
    # Prefix non-date columns to avoid collisions.
    rename = {c: f"breadth_{c}" for c in b.columns if c != "snapshot_date_kst" and not c.startswith("breadth_")}
    b = b.rename(columns=rename)
    for c in b.columns:
        if c != "snapshot_date_kst":
            b[c] = safe_float_series(b[c])
    return b


def filter_symbols(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["snapshot_date_kst", "symbol", "return_24h_pct"])

    if cfg.min_quote_volume > 0:
        out = out[(out["quote_volume"].fillna(0) >= cfg.min_quote_volume)]

    if cfg.drop_symbols_containing.strip():
        tokens = [x.strip() for x in cfg.drop_symbols_containing.split(",") if x.strip()]
        for t in tokens:
            out = out[~out["symbol"].str.contains(t, case=False, regex=False)]

    # Remove exact duplicates if scanner was run/merged multiple times.
    dedup_cols = ["snapshot_date_kst", "symbol"]
    if "side" in out.columns:
        dedup_cols.append("side")
    out = out.sort_values(["snapshot_date_kst", "symbol"]).drop_duplicates(dedup_cols, keep="last")
    return out


def compute_features(df: pd.DataFrame, breadth: Optional[pd.DataFrame]) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(["snapshot_date_kst", "symbol"]).reset_index(drop=True)

    # Cross-sectional ranking features for each KST 09:00 snapshot.
    gdate = d.groupby("snapshot_date_kst", observed=True)
    d["cs_rank_return_desc"] = gdate["return_24h_pct"].rank(method="average", ascending=False)
    d["cs_rank_return_asc"] = gdate["return_24h_pct"].rank(method="average", ascending=True)
    d["cs_rank_return_pct_desc"] = gdate["return_24h_pct"].rank(method="average", ascending=False, pct=True)
    d["cs_rank_volume_desc"] = gdate["quote_volume"].rank(method="average", ascending=False)
    d["cs_rank_volume_pct_desc"] = gdate["quote_volume"].rank(method="average", ascending=False, pct=True)

    d["is_up_top"] = (d["side"].astype(str).str.upper() == "UP_TOP").astype(int)
    d["is_down_top"] = (d["side"].astype(str).str.upper() == "DOWN_TOP").astype(int)
    d["side_direction"] = np.select(
        [d["is_up_top"].eq(1), d["is_down_top"].eq(1)],
        [1, -1],
        default=0,
    )

    d["abs_return_24h_pct"] = d["return_24h_pct"].abs()
    d["log_quote_volume"] = np.log1p(d["quote_volume"].clip(lower=0))
    d["signed_volume_pressure"] = d["return_24h_pct"] * d["log_quote_volume"]

    # Market-wide features from all rows in the selected data.
    market = gdate.agg(
        market_mean_return_pct=("return_24h_pct", "mean"),
        market_median_return_pct=("return_24h_pct", "median"),
        market_std_return_pct=("return_24h_pct", "std"),
        market_total_quote_volume=("quote_volume", "sum"),
        market_symbol_count=("symbol", "nunique"),
    ).reset_index()
    up_ratio = (d.assign(_is_up=(d["return_24h_pct"] > 0).astype(float))
                  .groupby("snapshot_date_kst", observed=True)["_is_up"]
                  .mean()
                  .rename("market_up_ratio")
                  .reset_index())
    market = market.merge(up_ratio, on="snapshot_date_kst", how="left")
    market["market_down_ratio"] = 1.0 - market["market_up_ratio"]
    market["market_log_total_quote_volume"] = np.log1p(market["market_total_quote_volume"].clip(lower=0))

    # BTC return as broad market anchor if present.
    btc = d.loc[d["symbol"].eq("BTCUSDT"), ["snapshot_date_kst", "return_24h_pct"]].rename(
        columns={"return_24h_pct": "btc_return_24h_pct"}
    )
    market = market.merge(btc, on="snapshot_date_kst", how="left")

    d = d.merge(market, on="snapshot_date_kst", how="left")

    b = normalize_breadth(breadth)
    if b is not None:
        d = d.merge(b, on="snapshot_date_kst", how="left")

    # Symbol-specific lag/rolling features. These use only earlier 09:00 snapshots.
    d = d.sort_values(["symbol", "snapshot_date_kst"]).reset_index(drop=True)
    gs = d.groupby("symbol", observed=True)
    for lag in [1, 2, 3, 5, 10]:
        d[f"symbol_return_lag{lag}_pct"] = gs["return_24h_pct"].shift(lag)
        d[f"symbol_volume_lag{lag}"] = gs["quote_volume"].shift(lag)

    for win in [3, 5, 10, 20]:
        shifted_ret = gs["return_24h_pct"].shift(1)
        d[f"symbol_return_roll{win}_mean_pct"] = shifted_ret.groupby(d["symbol"], observed=True).rolling(win, min_periods=2).mean().reset_index(level=0, drop=True)
        d[f"symbol_return_roll{win}_std_pct"] = shifted_ret.groupby(d["symbol"], observed=True).rolling(win, min_periods=2).std().reset_index(level=0, drop=True)
        shifted_vol = gs["quote_volume"].shift(1)
        d[f"symbol_volume_roll{win}_mean"] = shifted_vol.groupby(d["symbol"], observed=True).rolling(win, min_periods=2).mean().reset_index(level=0, drop=True)

    d["symbol_return_vs_market"] = d["return_24h_pct"] - d["market_mean_return_pct"]
    d["symbol_return_vs_btc"] = d["return_24h_pct"] - d["btc_return_24h_pct"]

    # Calendar features. KST 09:00 snapshots are one row per date.
    d["day_of_week"] = d["snapshot_date_kst"].dt.dayofweek.astype("Int64").astype(str)
    d["month"] = d["snapshot_date_kst"].dt.month.astype("Int64").astype(str)

    # Target. Keep latest no-target rows for prediction later.
    # The default target is positive next-24h return.
    # The command-line threshold is applied later in train_and_save.
    d["target_up_next_24h"] = (d["next_24h_return_pct"] > 0).astype(float)
    d.loc[d["next_24h_return_pct"].isna(), "target_up_next_24h"] = np.nan
    d["target_trade_return_long_pct"] = d["next_24h_return_pct"]
    d["target_trade_return_short_pct"] = -d["next_24h_return_pct"]

    d = d.sort_values(["snapshot_date_kst", "symbol"]).reset_index(drop=True)
    return d


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {
        "snapshot_date_kst",
        "snapshot_kst_9am",
        "open_time",
        "close_time",
        "next_24h_return_pct",
        "target_up_next_24h",
        "target_trade_return_long_pct",
        "target_trade_return_short_pct",
    }
    categorical = ["symbol", "side", "day_of_week", "month"]
    categorical = [c for c in categorical if c in df.columns]

    numeric = []
    for c in df.columns:
        if c in exclude or c in categorical:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            numeric.append(c)
    return numeric, categorical


def make_model(model_name: str, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_scaled = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    numeric_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", make_onehot_encoder()),
    ])

    if model_name == "logistic":
        pre = ColumnTransformer([
            ("num", numeric_scaled, numeric_cols),
            ("cat", categorical, categorical_cols),
        ])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_name == "random_forest":
        pre = ColumnTransformer([
            ("num", numeric_tree, numeric_cols),
            ("cat", categorical, categorical_cols),
        ])
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "dummy":
        pre = ColumnTransformer([
            ("num", numeric_tree, numeric_cols),
            ("cat", categorical, categorical_cols),
        ])
        clf = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("preprocess", pre), ("model", clf)])


def split_by_time(trainable: pd.DataFrame, test_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(trainable["snapshot_date_kst"].dropna().unique()))
    if len(dates) < 10:
        raise ValueError(f"Too few dates for time split: {len(dates)} dates")
    if test_days <= 0 or test_days >= len(dates):
        test_days = max(5, int(math.ceil(len(dates) * 0.2)))
    test_dates = set(dates[-test_days:])
    train = trainable[~trainable["snapshot_date_kst"].isin(test_dates)].copy()
    test = trainable[trainable["snapshot_date_kst"].isin(test_dates)].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced empty train or test set.")
    return train, test


def predict_proba_up(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    clf = model.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            # Dummy can have only one class if training target is one-sided.
            cls = getattr(clf, "classes_", np.array([0]))
            return np.ones(len(X)) if cls[0] == 1 else np.zeros(len(X))
        classes = getattr(clf, "classes_", np.array([0, 1]))
        idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
        return proba[:, idx]
    # Fallback to hard prediction.
    return model.predict(X).astype(float)


def evaluate_predictions(y_true: np.ndarray, prob_up: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob_up >= threshold).astype(int)
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    out["precision"] = float(precision_score(y_true, pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, pred, zero_division=0))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob_up))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["positive_rate_true"] = float(np.mean(y_true))
    out["positive_rate_pred"] = float(np.mean(pred))
    return out


def daily_long_short_backtest(pred_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    rows = []
    needed = ["snapshot_date_kst", "symbol", "prob_up", "next_24h_return_pct"]
    p = pred_df.dropna(subset=needed).copy()
    for dt, g in p.groupby("snapshot_date_kst", observed=True):
        g = g.sort_values("prob_up", ascending=False)
        longs = g.head(top_k)
        shorts = g.tail(top_k)
        long_ret = float(longs["next_24h_return_pct"].mean()) if len(longs) else np.nan
        short_ret = float((-shorts["next_24h_return_pct"]).mean()) if len(shorts) else np.nan
        ls_ret = np.nanmean([long_ret, short_ret])
        rows.append({
            "snapshot_date_kst": dt,
            "long_symbols": ",".join(longs["symbol"].astype(str).tolist()),
            "short_symbols": ",".join(shorts["symbol"].astype(str).tolist()),
            "long_avg_next_24h_return_pct": long_ret,
            "short_avg_next_24h_return_pct": short_ret,
            "long_short_avg_return_pct": float(ls_ret) if not np.isnan(ls_ret) else np.nan,
            "num_candidates": int(len(g)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["long_cum_return_gross"] = (1 + out["long_avg_next_24h_return_pct"].fillna(0) / 100.0).cumprod() - 1
        out["short_cum_return_gross"] = (1 + out["short_avg_next_24h_return_pct"].fillna(0) / 100.0).cumprod() - 1
        out["long_short_cum_return_gross"] = (1 + out["long_short_avg_return_pct"].fillna(0) / 100.0).cumprod() - 1
    return out


def baseline_rank_backtest(test_df: pd.DataFrame, top_k: int) -> Dict[str, float]:
    # Baseline 1: buy strongest current 24h returns.
    rows = []
    for dt, g in test_df.dropna(subset=["next_24h_return_pct"]).groupby("snapshot_date_kst", observed=True):
        strongest = g.sort_values("return_24h_pct", ascending=False).head(top_k)
        weakest = g.sort_values("return_24h_pct", ascending=True).head(top_k)
        rows.append({
            "momentum_long_avg_pct": strongest["next_24h_return_pct"].mean(),
            "reversal_long_weak_avg_pct": weakest["next_24h_return_pct"].mean(),
            "short_weak_avg_pct": (-weakest["next_24h_return_pct"]).mean(),
        })
    if not rows:
        return {}
    b = pd.DataFrame(rows)
    return {
        "baseline_buy_strongest_avg_pct": float(b["momentum_long_avg_pct"].mean()),
        "baseline_buy_weakest_reversal_avg_pct": float(b["reversal_long_weak_avg_pct"].mean()),
        "baseline_short_weakest_avg_pct": float(b["short_weak_avg_pct"].mean()),
    }


def feature_importance(model: Pipeline, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    try:
        pre = model.named_steps["preprocess"]
        names = pre.get_feature_names_out()
    except Exception:
        names = np.array(numeric_cols + categorical_cols)

    clf = model.named_steps["model"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        kind = "feature_importance"
    elif hasattr(clf, "coef_"):
        vals = clf.coef_.ravel()
        kind = "coefficient"
    else:
        return pd.DataFrame()

    m = min(len(names), len(vals))
    out = pd.DataFrame({"feature": names[:m], kind: vals[:m]})
    out["abs_value"] = out[kind].abs()
    return out.sort_values("abs_value", ascending=False)


def train_and_save(cfg: Config) -> None:
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw, breadth = load_input_files(cfg)
    raw = normalize_columns(raw)
    raw = filter_symbols(raw, cfg)
    dataset = compute_features(raw, breadth)
    if cfg.target_return_threshold != 0:
        dataset["target_up_next_24h"] = (dataset["next_24h_return_pct"] > cfg.target_return_threshold).astype(float)
        dataset.loc[dataset["next_24h_return_pct"].isna(), "target_up_next_24h"] = np.nan

    dataset_path = outdir / "training_dataset_with_features.csv"
    dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    trainable = dataset.dropna(subset=["target_up_next_24h"]).copy()
    trainable["target_up_next_24h"] = trainable["target_up_next_24h"].astype(int)
    if len(trainable) < 200:
        raise ValueError(
            f"Too few trainable rows: {len(trainable)}. "
            "Run the scanner with more days, e.g. --days 365 or --days 730."
        )

    train_df, test_df = split_by_time(trainable, cfg.test_days)
    numeric_cols, categorical_cols = get_feature_columns(dataset)

    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df["target_up_next_24h"].values
    X_test = test_df[numeric_cols + categorical_cols]
    y_test = test_df["target_up_next_24h"].values

    models_to_fit = [cfg.model] if cfg.model != "auto" else ["dummy", "logistic", "random_forest"]
    fitted: Dict[str, Pipeline] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name in models_to_fit:
        print(f"Training {name}...")
        pipe = make_model(name, numeric_cols, categorical_cols)
        pipe.fit(X_train, y_train)
        prob = predict_proba_up(pipe, X_test)
        m = evaluate_predictions(y_test, prob, cfg.probability_threshold)
        m.update({
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_start": str(train_df["snapshot_date_kst"].min().date()),
            "train_end": str(train_df["snapshot_date_kst"].max().date()),
            "test_start": str(test_df["snapshot_date_kst"].min().date()),
            "test_end": str(test_df["snapshot_date_kst"].max().date()),
            "num_numeric_features": int(len(numeric_cols)),
            "num_categorical_features": int(len(categorical_cols)),
        })
        fitted[name] = pipe
        metrics[name] = m

    # Choose by balanced accuracy, but do not pick dummy if tied unless it is truly best.
    best_name = max(metrics, key=lambda k: (metrics[k].get("balanced_accuracy", -1), metrics[k].get("roc_auc", -1)))
    best = fitted[best_name]
    print(f"Best model: {best_name}")

    test_pred = test_df.copy()
    test_pred["prob_up"] = predict_proba_up(best, test_df[numeric_cols + categorical_cols])
    test_pred["pred_up"] = (test_pred["prob_up"] >= cfg.probability_threshold).astype(int)
    test_pred["pred_signal"] = np.where(test_pred["pred_up"].eq(1), "LONG/up", "SHORT/down")
    test_pred.to_csv(outdir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    backtest = daily_long_short_backtest(test_pred, cfg.top_k)
    backtest.to_csv(outdir / "daily_model_long_short_backtest.csv", index=False, encoding="utf-8-sig")

    baseline = baseline_rank_backtest(test_df, cfg.top_k)
    if not backtest.empty:
        metrics[best_name]["model_long_avg_daily_pct"] = float(backtest["long_avg_next_24h_return_pct"].mean())
        metrics[best_name]["model_short_avg_daily_pct"] = float(backtest["short_avg_next_24h_return_pct"].mean())
        metrics[best_name]["model_long_short_avg_daily_pct"] = float(backtest["long_short_avg_return_pct"].mean())
        metrics[best_name]["model_long_short_final_cum_gross"] = float(backtest["long_short_cum_return_gross"].iloc[-1])
    metrics[best_name].update(baseline)

    latest = dataset[dataset["next_24h_return_pct"].isna()].copy()
    if latest.empty:
        # Fallback: latest date rows, even if target exists in historical backtest.
        max_date = dataset["snapshot_date_kst"].max()
        latest = dataset[dataset["snapshot_date_kst"].eq(max_date)].copy()
    if not latest.empty:
        latest["prob_up"] = predict_proba_up(best, latest[numeric_cols + categorical_cols])
        latest["pred_up"] = (latest["prob_up"] >= cfg.probability_threshold).astype(int)
        latest["pred_signal"] = np.where(latest["pred_up"].eq(1), "LONG/up", "SHORT/down")
        latest = latest.sort_values("prob_up", ascending=False)
        latest.to_csv(outdir / "latest_predictions.csv", index=False, encoding="utf-8-sig")
        latest.head(cfg.top_k).to_csv(outdir / "latest_model_long_candidates.csv", index=False, encoding="utf-8-sig")
        latest.tail(cfg.top_k).sort_values("prob_up", ascending=True).to_csv(outdir / "latest_model_short_candidates.csv", index=False, encoding="utf-8-sig")

    imp = feature_importance(best, numeric_cols, categorical_cols)
    if not imp.empty:
        imp.to_csv(outdir / "feature_importance_or_coefficients.csv", index=False, encoding="utf-8-sig")

    model_path = outdir / "trained_9am_rank_model.joblib"
    joblib.dump({
        "model_name": best_name,
        "pipeline": best,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "config": asdict(cfg),
    }, model_path)

    summary = {
        "best_model": best_name,
        "all_model_metrics": metrics,
        "best_model_metrics": metrics[best_name],
        "outputs": {
            "dataset": str(dataset_path),
            "test_predictions": str(outdir / "test_predictions.csv"),
            "latest_predictions": str(outdir / "latest_predictions.csv"),
            "latest_long_candidates": str(outdir / "latest_model_long_candidates.csv"),
            "latest_short_candidates": str(outdir / "latest_model_short_candidates.csv"),
            "daily_backtest": str(outdir / "daily_model_long_short_backtest.csv"),
            "model": str(model_path),
        },
    }
    with open(outdir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Saved metrics: {outdir / 'model_metrics.json'}")
    print(f"Saved latest predictions: {outdir / 'latest_predictions.csv'}")
    print(f"Saved model: {model_path}")
    print("\nBest model metrics:")
    for k, v in metrics[best_name].items():
        print(f"  {k}: {v}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train a model using Binance KST 09:00 ranking scanner outputs.")
    p.add_argument("--data-dir", default="outputs_binance_9am_rank", help="Folder containing scanner CSV outputs.")
    p.add_argument("--all-returns", default=None, help="Path to all_returns_by_9am.csv. Overrides --data-dir.")
    p.add_argument("--top-bottom", default=None, help="Path to top_bottom_by_9am.csv. Overrides --data-dir.")
    p.add_argument("--market-breadth", default=None, help="Path to market_breadth_by_9am.csv. Overrides --data-dir.")
    p.add_argument("--sample-mode", choices=["all", "top_bottom"], default="all", help="Train on all symbols or only top/bottom ranking rows.")
    p.add_argument("--outdir", default="outputs_9am_rank_model", help="Output folder for model results.")
    p.add_argument("--min-quote-volume", type=float, default=0.0, help="Minimum 24h quote volume filter in USDT.")
    p.add_argument("--test-days", type=int, default=60, help="Last N dates used as test period.")
    p.add_argument("--top-k", type=int, default=10, help="Number of long/short candidates per day in model backtest.")
    p.add_argument("--probability-threshold", type=float, default=0.5, help="Probability threshold for up/down classification.")
    p.add_argument("--model", choices=["auto", "dummy", "logistic", "random_forest"], default="auto", help="Model type.")
    p.add_argument("--target-return-threshold", type=float, default=0.0, help="Reserved. Target is next_24h_return_pct > threshold.")
    p.add_argument("--drop-symbols-containing", default="", help="Comma-separated tokens to exclude from symbols, e.g. USDC,BTCDOM.")
    a = p.parse_args()
    return Config(**vars(a))


if __name__ == "__main__":
    cfg = parse_args()
    train_and_save(cfg)
