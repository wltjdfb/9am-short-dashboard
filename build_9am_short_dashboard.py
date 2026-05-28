"""Build a local, standalone HTML dashboard for the Binance 9AM short-only model.

PowerShell example:
  python build_9am_short_dashboard.py --model-dir outputs_9am_rank_model_365_vol100m_test180 --top-n 3 --short-threshold 0.45 --cost-pct 0.50 --outdir dashboard_9am_short

Then open:
  dashboard_9am_short\\index.html

This script reads files produced by:
  - train_binance_9am_rank_model.py
  - audit_9am_rank_sides.py

Required file:
  <model-dir>/latest_predictions.csv

Optional files:
  <model-dir>/model_metrics.json
  <model-dir>/audit_sides_top3_cost0.5.json
  <model-dir>/audit_short_top3_cost0.5_thr0.45.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DANGER_NOTE = "리서치/페이퍼트레이딩용입니다. 레버리지, 청산, 슬리피지, 펀딩비는 별도 검증이 필요합니다."


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    # utf-8-sig handles CSVs saved with BOM on Windows.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return default
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return default
    try:
        v = float(s)
    except ValueError:
        return default
    return v if math.isfinite(v) else default


def fmt_pct(v: Any, digits: int = 2, signed: bool = True) -> str:
    x = to_float(v)
    if x is None:
        return "—"
    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x:.{digits}f}%"


def fmt_prob(v: Any, digits: int = 1) -> str:
    x = to_float(v)
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}%"


def fmt_num(v: Any, digits: int = 2) -> str:
    x = to_float(v)
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def fmt_int(v: Any) -> str:
    x = to_float(v)
    if x is None:
        return "—"
    return f"{int(round(x)):,}"


def fmt_volume(v: Any) -> str:
    x = to_float(v)
    if x is None:
        return "—"
    if abs(x) >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}K"
    return f"{x:.0f}"


def css_class_for_pct(v: Any) -> str:
    x = to_float(v, 0.0) or 0.0
    if x > 0:
        return "pos"
    if x < 0:
        return "neg"
    return "muted"


def safe_get(row: Dict[str, Any], key: str, fallback: str = "") -> str:
    v = row.get(key, fallback)
    return "" if v is None else str(v)


def pick_latest_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return []
    dates = [safe_get(r, "snapshot_date_kst") for r in rows if safe_get(r, "snapshot_date_kst")]
    if not dates:
        return rows
    latest = max(dates)
    return [r for r in rows if safe_get(r, "snapshot_date_kst") == latest]


def sort_by_float(rows: List[Dict[str, str]], key: str, reverse: bool = False) -> List[Dict[str, str]]:
    return sorted(rows, key=lambda r: (to_float(r.get(key), float("inf")) if not reverse else -(to_float(r.get(key), -float("inf")) or -float("inf"))))


def load_latest_predictions(model_dir: Path, min_quote_volume: Optional[float]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    path = model_dir / "latest_predictions.csv"
    rows = read_csv_rows(path)
    latest = pick_latest_rows(rows)
    if min_quote_volume is not None and min_quote_volume > 0:
        latest = [r for r in latest if (to_float(r.get("quote_volume"), 0.0) or 0.0) >= min_quote_volume]
    meta = {
        "path": str(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "latest_count": len(latest),
        "snapshot_date": safe_get(latest[0], "snapshot_date_kst") if latest else "",
        "snapshot_time": safe_get(latest[0], "snapshot_kst_9am") if latest else "",
    }
    return latest, meta


def load_model_metrics(model_dir: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    path = model_dir / "model_metrics.json"
    data = read_json(path)
    return data, {"path": str(path), "exists": path.exists()}


def find_audit_json(model_dir: Path, top_n: int, cost_pct: float) -> Optional[Path]:
    exact = model_dir / f"audit_sides_top{top_n}_cost{cost_pct}.json"
    if exact.exists():
        return exact
    # If user typed 0.50, Python float prints 0.5 in audit script names.
    exact2 = model_dir / f"audit_sides_top{top_n}_cost{float(cost_pct)}.json"
    if exact2.exists():
        return exact2
    candidates = [
        model_dir / "audit_sides_top3_cost0.5.json",
        model_dir / "audit_sides_top3_cost0.2.json",
        model_dir / "audit_sides_top5_cost0.5.json",
        model_dir / "audit_sides_top5_cost0.2.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    found = sorted(model_dir.glob("audit_sides_top*_cost*.json"))
    return found[0] if found else None


def close_enough(a: Any, b: Any, eps: float = 1e-9) -> bool:
    aa = to_float(a)
    bb = to_float(b)
    if aa is None or bb is None:
        return a is None and b is None
    return abs(aa - bb) <= eps


def load_audit(model_dir: Path, top_n: int, cost_pct: float, short_threshold: float) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    path = find_audit_json(model_dir, top_n, cost_pct)
    data = read_json(path) if path else None
    status = {"path": str(path) if path else "", "exists": bool(path and path.exists())}
    short_item = None
    long_item = None
    if isinstance(data, dict):
        for item in data.get("audits", []):
            if item.get("side") == "SHORT" and close_enough(item.get("threshold"), short_threshold):
                short_item = item
            if item.get("side") == "LONG" and close_enough(item.get("threshold"), 0.55):
                long_item = item
    return data, status, short_item, long_item


def load_daily_returns(item: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], Optional[Path]]:
    if not item:
        return [], None
    p = item.get("daily_file")
    if not p:
        return [], None
    path = Path(str(p))
    # audit json may contain a relative path from the project root.
    rows = read_csv_rows(path)
    return rows, path if path.exists() else None


def make_sparkline_svg(values: List[float], width: int = 680, height: int = 170) -> str:
    if len(values) < 2:
        return '<div class="empty-chart">차트를 만들 일별 수익률 데이터가 부족합니다.</div>'
    # Build cumulative curve from daily percentage returns.
    curve = []
    eq = 1.0
    for r in values:
        if r is None or not math.isfinite(r):
            continue
        eq *= (1.0 + r / 100.0)
        curve.append(eq)
    if len(curve) < 2:
        return '<div class="empty-chart">차트를 만들 일별 수익률 데이터가 부족합니다.</div>'
    mn, mx = min(curve), max(curve)
    if abs(mx - mn) < 1e-12:
        mn -= 1
        mx += 1
    pad_x = 14
    pad_y = 18
    pts = []
    for i, y in enumerate(curve):
        x = pad_x + i * (width - 2 * pad_x) / (len(curve) - 1)
        yy = height - pad_y - (y - mn) * (height - 2 * pad_y) / (mx - mn)
        pts.append(f"{x:.2f},{yy:.2f}")
    last = curve[-1]
    first = curve[0]
    label = f"{last:.2f}x" if last >= 1 else f"{last:.3f}x"
    return f'''
<svg class="equity-svg" viewBox="0 0 {width} {height}" role="img" aria-label="누적 성과 곡선">
  <defs>
    <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="rgba(52,211,153,0.42)"/>
      <stop offset="100%" stop-color="rgba(52,211,153,0.02)"/>
    </linearGradient>
  </defs>
  <line x1="{pad_x}" y1="{height-pad_y}" x2="{width-pad_x}" y2="{height-pad_y}" class="axis"/>
  <polyline points="{escape(' '.join(pts))}" class="eq-line" fill="none"/>
  <polygon points="{pad_x},{height-pad_y} {escape(' '.join(pts))} {width-pad_x},{height-pad_y}" fill="url(#eqFill)"/>
  <text x="{width-pad_x-5}" y="22" text-anchor="end" class="chart-label">{escape(label)}</text>
</svg>'''


def make_return_bars(values: List[float], max_bars: int = 80) -> str:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return '<div class="empty-chart">일별 수익률 막대 데이터가 없습니다.</div>'
    if len(vals) > max_bars:
        # sample recent bars for readability
        vals = vals[-max_bars:]
    max_abs = max(abs(v) for v in vals) or 1.0
    bars = []
    for v in vals:
        h = max(4, min(100, abs(v) / max_abs * 100))
        cls = "bar-pos" if v >= 0 else "bar-neg"
        bars.append(f'<span class="mini-bar {cls}" style="height:{h:.1f}%" title="{v:.2f}%"></span>')
    return '<div class="bar-wrap">' + ''.join(bars) + '</div>'


def pct_width_from_prob_down(prob_up: Any) -> float:
    p = to_float(prob_up)
    if p is None:
        return 0.0
    return max(0.0, min(100.0, (1.0 - p) * 100.0))


def build_candidate_cards(shorts: List[Dict[str, str]], threshold: float, top_n: int) -> str:
    if not shorts:
        return '<div class="empty-state">오늘은 조건을 통과한 숏 후보가 없습니다. 억지로 종목 수를 채우지 않는 것이 룰입니다.</div>'
    cards = []
    for idx, r in enumerate(shorts, start=1):
        symbol = safe_get(r, "symbol")
        prob_up = to_float(r.get("prob_up"), 0.0) or 0.0
        prob_down = 1.0 - prob_up
        ret24 = to_float(r.get("return_24h_pct"))
        quote = to_float(r.get("quote_volume"))
        high = to_float(r.get("high_from_open_pct"))
        low = to_float(r.get("low_from_open_pct"))
        range_pct = to_float(r.get("range_pct"))
        taker = to_float(r.get("taker_buy_quote_share"))
        binance_url = f"https://www.binance.com/en/futures/{symbol}" if symbol else "#"
        badge = "통과" if prob_up <= threshold else "관찰"
        cards.append(f'''
<article class="candidate-card">
  <div class="candidate-head">
    <div>
      <div class="rank-badge">SHORT #{idx}</div>
      <h3>{escape(symbol)}</h3>
    </div>
    <span class="signal-pill">{escape(badge)}</span>
  </div>
  <div class="prob-row">
    <span>하락 확률 추정</span>
    <strong>{prob_down * 100:.1f}%</strong>
  </div>
  <div class="prob-track"><div class="prob-fill" style="width:{pct_width_from_prob_down(prob_up):.1f}%"></div></div>
  <div class="candidate-grid">
    <div><small>prob_up</small><b>{fmt_prob(prob_up)}</b></div>
    <div><small>24h 수익률</small><b class="{css_class_for_pct(ret24)}">{fmt_pct(ret24)}</b></div>
    <div><small>거래대금</small><b>{fmt_volume(quote)} USDT</b></div>
    <div><small>당일 변동폭</small><b>{fmt_pct(range_pct, signed=False)}</b></div>
    <div><small>고가/시가</small><b class="{css_class_for_pct(high)}">{fmt_pct(high)}</b></div>
    <div><small>저가/시가</small><b class="{css_class_for_pct(low)}">{fmt_pct(low)}</b></div>
  </div>
  <div class="candidate-foot">
    <span>테이커 매수 비중 {fmt_prob(taker)}</span>
    <a href="{escape(binance_url)}" target="_blank" rel="noreferrer">Binance 열기</a>
  </div>
</article>''')
    note = "" if len(shorts) >= top_n else f'<div class="rule-note">오늘은 기준을 통과한 후보가 {len(shorts)}개입니다. 룰상 최대 {top_n}개지만, 부족하면 억지로 채우지 않습니다.</div>'
    return note + '<div class="candidate-list">' + ''.join(cards) + '</div>'


def build_table(rows: List[Dict[str, str]], columns: List[Tuple[str, str, str]], max_rows: int = 30, table_id: str = "dataTable") -> str:
    if not rows:
        return '<div class="empty-state">표시할 데이터가 없습니다.</div>'
    head = ''.join(f'<th>{escape(label)}</th>' for _, label, _ in columns)
    body = []
    for r in rows[:max_rows]:
        tds = []
        for key, _label, kind in columns:
            val = r.get(key)
            cls = ""
            if kind == "pct":
                text = fmt_pct(val)
                cls = css_class_for_pct(val)
            elif kind == "prob":
                text = fmt_prob(val)
            elif kind == "vol":
                text = fmt_volume(val)
            elif kind == "int":
                text = fmt_int(val)
            else:
                text = safe_get(r, key, "—") or "—"
            tds.append(f'<td class="{cls}">{escape(text)}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    return f'<div class="table-scroll"><table id="{escape(table_id)}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def metric_card(title: str, value: str, sub: str = "", cls: str = "") -> str:
    return f'''
<div class="metric-card {escape(cls)}">
  <small>{escape(title)}</small>
  <strong>{escape(value)}</strong>
  <span>{escape(sub)}</span>
</div>'''


def build_file_status(statuses: List[Dict[str, Any]]) -> str:
    items = []
    for s in statuses:
        exists = bool(s.get("exists"))
        items.append(f'<li class="{"ok" if exists else "bad"}"><b>{escape(s.get("label", "file"))}</b><span>{escape(s.get("path", ""))}</span></li>')
    return '<ul class="file-status">' + ''.join(items) + '</ul>'


def build_html(
    model_dir: Path,
    outdir: Path,
    latest_rows: List[Dict[str, str]],
    latest_meta: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    metrics_status: Dict[str, Any],
    audit_data: Optional[Dict[str, Any]],
    audit_status: Dict[str, Any],
    short_audit: Optional[Dict[str, Any]],
    long_audit: Optional[Dict[str, Any]],
    daily_rows: List[Dict[str, str]],
    top_n: int,
    short_threshold: float,
    cost_pct: float,
    min_quote_volume: Optional[float],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # latest rows already filtered to latest snapshot. Sort all variants.
    short_candidates_all = [r for r in latest_rows if (to_float(r.get("prob_up"), 1.0) or 1.0) <= short_threshold]
    short_candidates = sorted(short_candidates_all, key=lambda r: to_float(r.get("prob_up"), 1.0) or 1.0)[:top_n]
    near_candidates = [r for r in latest_rows if short_threshold < (to_float(r.get("prob_up"), 1.0) or 1.0) <= min(0.50, short_threshold + 0.08)]
    near_candidates = sorted(near_candidates, key=lambda r: to_float(r.get("prob_up"), 1.0) or 1.0)[:10]
    ignored_longs = sorted([r for r in latest_rows if (to_float(r.get("prob_up"), 0.0) or 0.0) >= 0.55], key=lambda r: to_float(r.get("prob_up"), 0.0) or 0.0, reverse=True)[:10]

    first = latest_rows[0] if latest_rows else {}
    market_up_ratio = to_float(first.get("market_up_ratio"))
    market_down_ratio = to_float(first.get("market_down_ratio"))
    market_mean = to_float(first.get("market_mean_return_pct"))
    btc_ret = to_float(first.get("btc_return_24h_pct"))
    market_symbols = first.get("market_symbol_count") or first.get("breadth_symbols_count")
    snapshot = latest_meta.get("snapshot_time") or latest_meta.get("snapshot_date") or "—"

    best = metrics.get("best_model_metrics", {}) if isinstance(metrics, dict) else {}
    best_name = metrics.get("best_model", "—") if isinstance(metrics, dict) else "—"
    train_span = ""
    if best:
        train_span = f"train {best.get('train_start','?')}~{best.get('train_end','?')} / test {best.get('test_start','?')}~{best.get('test_end','?')}"

    auc = to_float(best.get("roc_auc"))
    bal = to_float(best.get("balanced_accuracy"))
    acc = to_float(best.get("accuracy"))
    prec = to_float(best.get("precision"))
    rec = to_float(best.get("recall"))

    daily_values = []
    for r in daily_rows:
        v = to_float(r.get("daily_return_pct_net"))
        if v is not None:
            daily_values.append(v)

    audit_title = "SHORT audit"
    if short_audit:
        audit_title = f"SHORT ≤ {short_threshold:.2f}, top {top_n}, cost {cost_pct:.2f}%"

    short_avg = short_audit.get("avg_daily_pct_net") if short_audit else None
    short_med = short_audit.get("median_daily_pct_net") if short_audit else None
    short_win = short_audit.get("daily_win_rate") if short_audit else None
    short_worst = short_audit.get("worst_day_pct_net") if short_audit else None
    short_gross = short_audit.get("gross_compound_multiple_net") if short_audit else None
    short_days = short_audit.get("days") if short_audit else None
    top5_share = short_audit.get("top5_days_share_of_sum") if short_audit else None

    cards = ''.join([
        metric_card("오늘 숏 후보", f"{len(short_candidates)}개", f"기준 prob_up ≤ {short_threshold:.2f}, 최대 {top_n}개", "accent"),
        metric_card("시장 상승 비율", fmt_prob(market_up_ratio), f"하락 비율 {fmt_prob(market_down_ratio)}", ""),
        metric_card("시장 평균 수익률", fmt_pct(market_mean), f"BTC {fmt_pct(btc_ret)}", css_class_for_pct(market_mean)),
        metric_card("검증 평균", fmt_pct(short_avg), f"{audit_title}", css_class_for_pct(short_avg)),
        metric_card("검증 승률", fmt_prob(short_win), f"days {short_days if short_days is not None else '—'}", ""),
        metric_card("최악일", fmt_pct(short_worst), "레버리지 사용 시 가장 중요", "danger"),
    ])

    metric_grid = ''.join([
        metric_card("Best model", str(best_name), train_span, ""),
        metric_card("ROC AUC", fmt_num(auc, 4), "0.50 이상이면 랜덤보다 우위", ""),
        metric_card("Balanced Acc", fmt_num(bal, 4), "상승/하락 균형 정확도", ""),
        metric_card("Accuracy", fmt_num(acc, 4), "전체 정답률", ""),
        metric_card("Precision", fmt_num(prec, 4), "상승 예측 정밀도", ""),
        metric_card("Recall", fmt_num(rec, 4), "상승 포착률", ""),
    ])

    audit_grid = ''.join([
        metric_card("평균 일수익", fmt_pct(short_avg), audit_title, css_class_for_pct(short_avg)),
        metric_card("중앙값", fmt_pct(short_med), "평균보다 안정성 확인", css_class_for_pct(short_med)),
        metric_card("승률", fmt_prob(short_win), "일 단위 기준", ""),
        metric_card("최악일", fmt_pct(short_worst), "청산 리스크 체크", "danger"),
        metric_card("상위 5일 의존도", fmt_num(top5_share, 4), "낮을수록 특정 날 의존↓", ""),
        metric_card("누적 배수", fmt_num(short_gross, 2) + "x" if short_gross is not None else "—", "비용 차감 후 gross", "accent"),
    ])

    short_table = build_table(short_candidates, [
        ("symbol", "종목", "text"),
        ("prob_up", "prob_up", "prob"),
        ("return_24h_pct", "24h", "pct"),
        ("quote_volume", "거래대금", "vol"),
        ("high_from_open_pct", "고가/시가", "pct"),
        ("low_from_open_pct", "저가/시가", "pct"),
        ("range_pct", "변동폭", "pct"),
    ], max_rows=top_n, table_id="shortTable")

    near_table = build_table(near_candidates, [
        ("symbol", "종목", "text"),
        ("prob_up", "prob_up", "prob"),
        ("return_24h_pct", "24h", "pct"),
        ("quote_volume", "거래대금", "vol"),
        ("pred_signal", "신호", "text"),
    ], max_rows=10, table_id="nearTable")

    ignored_table = build_table(ignored_longs, [
        ("symbol", "종목", "text"),
        ("prob_up", "prob_up", "prob"),
        ("return_24h_pct", "24h", "pct"),
        ("quote_volume", "거래대금", "vol"),
        ("pred_signal", "출력 신호", "text"),
    ], max_rows=10, table_id="longTable")

    daily_svg = make_sparkline_svg(daily_values)
    daily_bars = make_return_bars(daily_values)

    file_status = build_file_status([
        {"label": "latest_predictions", **latest_meta},
        {"label": "model_metrics", **metrics_status},
        {"label": "audit_sides", **audit_status},
    ])

    rules = f"""
<ol class="rule-list">
  <li><b>방향:</b> SHORT only. 롱 신호는 현재 검증상 사용하지 않습니다.</li>
  <li><b>후보:</b> prob_up ≤ {short_threshold:.2f} 종목만, 낮은 순서로 최대 {top_n}개.</li>
  <li><b>부족하면:</b> {top_n}개를 억지로 채우지 않습니다.</li>
  <li><b>시간:</b> KST 09:00 기준 진입 후보, 다음날 KST 09:00 기준 성과 검증.</li>
  <li><b>거래대금:</b> {fmt_volume(min_quote_volume)} USDT 이상 필터 {'' if min_quote_volume else '(모델 폴더 기준)'}</li>
</ol>
"""

    # Commands shown inside dashboard for daily refresh.
    cmd_build = f"python build_9am_short_dashboard.py --model-dir {model_dir} --top-n {top_n} --short-threshold {short_threshold:.2f} --cost-pct {cost_pct:.2f} --outdir {outdir}"
    cmd_audit = f"python audit_9am_rank_sides.py --dirs {model_dir} --top-n {top_n} --cost-pct {cost_pct:.2f} --include-no-threshold"

    html = f'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Binance 9AM SHORT Dashboard</title>
  <style>
    :root {{
      --bg: #07111f;
      --panel: rgba(15, 23, 42, .92);
      --panel2: rgba(30, 41, 59, .72);
      --line: rgba(148, 163, 184, .18);
      --text: #e5e7eb;
      --muted: #94a3b8;
      --green: #34d399;
      --red: #fb7185;
      --yellow: #fbbf24;
      --blue: #60a5fa;
      --purple: #a78bfa;
      --shadow: 0 20px 60px rgba(0,0,0,.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 20% 0%, rgba(96,165,250,.22), transparent 28%),
        radial-gradient(circle at 80% 10%, rgba(167,139,250,.18), transparent 26%),
        linear-gradient(180deg, #07111f 0%, #0f172a 100%);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
    }}
    a {{ color: #93c5fd; text-decoration: none; }}
    .shell {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .hero {{ display: grid; grid-template-columns: 1.4fr .8fr; gap: 18px; align-items: stretch; }}
    .hero-card, .panel, .metric-card, .candidate-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      border-radius: 22px;
    }}
    .hero-card {{ padding: 26px; position: relative; overflow: hidden; }}
    .hero-card::after {{ content:""; position:absolute; inset:auto -80px -120px auto; width:260px; height:260px; border-radius:50%; background:rgba(52,211,153,.12); }}
    .eyebrow {{ color: var(--green); font-weight: 800; letter-spacing: .08em; text-transform: uppercase; font-size: 12px; }}
    h1 {{ font-size: clamp(30px, 5vw, 54px); line-height: 1.03; margin: 12px 0 10px; letter-spacing: -0.05em; }}
    .subtitle {{ color: var(--muted); font-size: 16px; line-height: 1.6; max-width: 850px; }}
    .hero-side {{ padding: 22px; display:flex; flex-direction:column; justify-content:space-between; }}
    .snapshot {{ font-size: 24px; font-weight: 900; }}
    .small-muted {{ color: var(--muted); font-size: 13px; line-height:1.5; }}
    .pill-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
    .pill {{ border:1px solid var(--line); border-radius:999px; padding:8px 11px; background:rgba(15,23,42,.55); color:var(--muted); font-size:13px; }}
    .pill strong {{ color:var(--text); }}
    .grid-metrics {{ display:grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 16px 0 18px; }}
    .metric-card {{ padding: 16px; min-height: 116px; }}
    .metric-card small {{ display:block; color:var(--muted); font-size:12px; }}
    .metric-card strong {{ display:block; margin-top:9px; font-size:25px; letter-spacing:-.03em; }}
    .metric-card span {{ display:block; margin-top:8px; color:var(--muted); font-size:12px; line-height:1.4; }}
    .metric-card.accent {{ border-color:rgba(52,211,153,.38); background:linear-gradient(180deg, rgba(52,211,153,.14), rgba(15,23,42,.92)); }}
    .metric-card.danger {{ border-color:rgba(251,113,133,.38); background:linear-gradient(180deg, rgba(251,113,133,.14), rgba(15,23,42,.92)); }}
    .metric-card.pos strong, .pos {{ color: var(--green); }}
    .metric-card.neg strong, .neg {{ color: var(--red); }}
    .muted {{ color: var(--muted); }}
    .section-title {{ display:flex; align-items:end; justify-content:space-between; gap:12px; margin:28px 0 12px; }}
    .section-title h2 {{ margin:0; font-size:24px; letter-spacing:-.03em; }}
    .section-title p {{ margin:0; color:var(--muted); font-size:13px; }}
    .candidate-list {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .candidate-card {{ padding: 18px; }}
    .candidate-head {{ display:flex; justify-content:space-between; gap:12px; align-items:start; }}
    .candidate-card h3 {{ margin: 4px 0 0; font-size: 30px; letter-spacing:-.04em; }}
    .rank-badge {{ color:var(--green); font-size:12px; font-weight:900; }}
    .signal-pill {{ background:rgba(251,113,133,.14); color:#fecdd3; border:1px solid rgba(251,113,133,.28); padding:7px 10px; border-radius:999px; font-size:12px; font-weight:800; }}
    .prob-row {{ display:flex; justify-content:space-between; margin-top:20px; color:var(--muted); }}
    .prob-row strong {{ color:var(--text); font-size:22px; }}
    .prob-track {{ height:10px; border-radius:999px; background:rgba(148,163,184,.16); overflow:hidden; margin-top:8px; }}
    .prob-fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg, var(--red), var(--yellow)); }}
    .candidate-grid {{ display:grid; grid-template-columns: repeat(3,1fr); gap:10px; margin-top:16px; }}
    .candidate-grid div {{ background:rgba(15,23,42,.75); border:1px solid var(--line); border-radius:14px; padding:10px; }}
    .candidate-grid small {{ display:block; color:var(--muted); font-size:11px; }}
    .candidate-grid b {{ display:block; margin-top:5px; font-size:14px; }}
    .candidate-foot {{ display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:14px; color:var(--muted); font-size:12px; }}
    .panel {{ padding: 18px; margin-top: 14px; }}
    .two-col {{ display:grid; grid-template-columns: 1fr 1fr; gap:14px; }}
    .table-scroll {{ overflow-x:auto; border-radius:16px; border:1px solid var(--line); }}
    table {{ width:100%; border-collapse:collapse; min-width: 680px; background:rgba(15,23,42,.45); }}
    th, td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; font-size:13px; }}
    th {{ color:#cbd5e1; background:rgba(30,41,59,.85); font-size:12px; position:sticky; top:0; }}
    tr:hover td {{ background:rgba(96,165,250,.06); }}
    .rule-note, .empty-state, .warning {{ border:1px solid rgba(251,191,36,.32); background:rgba(251,191,36,.08); color:#fde68a; border-radius:16px; padding:13px 14px; margin-bottom:12px; line-height:1.55; }}
    .empty-state {{ color:var(--muted); border-color:var(--line); background:rgba(15,23,42,.55); }}
    .rule-list {{ margin:0; padding-left:20px; color:#cbd5e1; line-height:1.8; }}
    .equity-svg {{ width:100%; height:auto; display:block; }}
    .axis {{ stroke:rgba(148,163,184,.25); }}
    .eq-line {{ stroke:var(--green); stroke-width:3; filter: drop-shadow(0 0 10px rgba(52,211,153,.25)); }}
    .chart-label {{ fill:#d1fae5; font-size:18px; font-weight:900; }}
    .bar-wrap {{ height:130px; display:flex; align-items:flex-end; gap:3px; padding: 12px 4px 0; border-bottom:1px solid rgba(148,163,184,.2); }}
    .mini-bar {{ flex:1; min-width:3px; border-radius: 5px 5px 0 0; opacity:.85; }}
    .bar-pos {{ background:rgba(52,211,153,.72); }}
    .bar-neg {{ background:rgba(251,113,133,.72); }}
    .empty-chart {{ color:var(--muted); padding:40px 0; text-align:center; }}
    .file-status {{ list-style:none; padding:0; margin:0; display:grid; gap:10px; }}
    .file-status li {{ display:grid; grid-template-columns:160px 1fr; gap:10px; padding:10px; border-radius:12px; border:1px solid var(--line); background:rgba(15,23,42,.45); }}
    .file-status li b::before {{ content:"● "; }}
    .file-status li.ok b {{ color:var(--green); }}
    .file-status li.bad b {{ color:var(--red); }}
    .file-status span {{ color:var(--muted); font-size:12px; word-break:break-all; }}
    .code {{ background:#020617; border:1px solid var(--line); border-radius:14px; padding:12px; color:#bfdbfe; overflow:auto; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size:12px; }}
    .footer {{ margin:26px 0 8px; color:var(--muted); font-size:12px; text-align:center; }}
    .toolbar {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }}
    .toolbar input {{ width: 260px; max-width:100%; background:#020617; color:var(--text); border:1px solid var(--line); border-radius:12px; padding:10px 12px; }}
    .toolbar button {{ background:rgba(96,165,250,.16); color:#dbeafe; border:1px solid rgba(96,165,250,.28); border-radius:12px; padding:10px 12px; cursor:pointer; }}
    @media (max-width: 1180px) {{ .grid-metrics {{ grid-template-columns: repeat(3, 1fr); }} .candidate-list {{ grid-template-columns: 1fr; }} .hero {{ grid-template-columns:1fr; }} }}
    @media (max-width: 720px) {{ .shell {{ padding:14px; }} .grid-metrics {{ grid-template-columns: repeat(2, 1fr); }} .two-col {{ grid-template-columns:1fr; }} .candidate-grid {{ grid-template-columns: repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-card">
        <div class="eyebrow">Binance 9AM SHORT-only model</div>
        <h1>오전 9시 숏 후보 대시보드</h1>
        <p class="subtitle">거래대금 큰 종목 중 <b>prob_up ≤ {short_threshold:.2f}</b>인 약세 후보만 추립니다. 현재 검증 기준에서는 롱은 버리고, 숏 후보만 보는 구조입니다.</p>
        <div class="pill-row">
          <span class="pill">모델 폴더 <strong>{escape(str(model_dir))}</strong></span>
          <span class="pill">Top N <strong>{top_n}</strong></span>
          <span class="pill">Cost <strong>{cost_pct:.2f}%</strong></span>
          <span class="pill">생성 <strong>{escape(now)}</strong></span>
        </div>
      </div>
      <div class="hero-card hero-side">
        <div>
          <div class="small-muted">현재 스냅샷</div>
          <div class="snapshot">{escape(str(snapshot))}</div>
        </div>
        <div class="small-muted">{escape(DANGER_NOTE)}</div>
      </div>
    </section>

    <section class="grid-metrics">{cards}</section>

    <div class="warning"><b>운용 룰:</b> 이 화면은 매수 추천이 아니라 모델 리서치용입니다. 특히 숏은 24시간 종가가 유리해도 중간 급등으로 청산될 수 있으므로 레버리지는 별도 제한이 필요합니다.</div>

    <section class="section-title">
      <div><h2>오늘 숏 후보</h2><p>prob_up 낮은 순서. 조건 미충족 종목은 넣지 않습니다.</p></div>
    </section>
    {build_candidate_cards(short_candidates, short_threshold, top_n)}

    <section class="two-col">
      <div class="panel">
        <div class="section-title" style="margin-top:0"><div><h2>관찰 후보</h2><p>기준에는 못 미치지만 가까운 종목</p></div></div>
        {near_table}
      </div>
      <div class="panel">
        <div class="section-title" style="margin-top:0"><div><h2>롱 출력은 무시</h2><p>검증상 롱은 폐기한 상태</p></div></div>
        {ignored_table}
      </div>
    </section>

    <section class="panel">
      <div class="section-title" style="margin-top:0"><div><h2>숏 전용 검증 결과</h2><p>{escape(audit_title)}</p></div></div>
      <div class="grid-metrics">{audit_grid}</div>
      <div class="two-col">
        <div>
          <h3>누적 성과 곡선</h3>
          {daily_svg}
        </div>
        <div>
          <h3>최근 일별 수익률</h3>
          {daily_bars}
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="section-title" style="margin-top:0"><div><h2>모델 분류 성능</h2><p>수익률만 보지 말고 AUC와 Balanced Accuracy도 같이 봅니다.</p></div></div>
      <div class="grid-metrics">{metric_grid}</div>
    </section>

    <section class="two-col">
      <div class="panel">
        <div class="section-title" style="margin-top:0"><div><h2>룰 요약</h2><p>오늘 후보를 해석하는 기준</p></div></div>
        {rules}
      </div>
      <div class="panel">
        <div class="section-title" style="margin-top:0"><div><h2>데이터 파일 상태</h2><p>누락이면 해당 영역이 비어 보입니다.</p></div></div>
        {file_status}
      </div>
    </section>

    <section class="panel">
      <div class="section-title" style="margin-top:0"><div><h2>매일 갱신 명령</h2><p>모델 파일이 갱신된 뒤 이 명령으로 웹페이지를 다시 만듭니다.</p></div></div>
      <div class="code">{escape(cmd_audit)}<br>{escape(cmd_build)}</div>
    </section>

    <div class="footer">Generated by build_9am_short_dashboard.py · {escape(DANGER_NOTE)}</div>
  </main>
  <script>
    function filterTable(inputId, tableId) {{
      const input = document.getElementById(inputId);
      const table = document.getElementById(tableId);
      if (!input || !table) return;
      const q = input.value.toLowerCase();
      for (const row of table.querySelectorAll('tbody tr')) {{
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
      }}
    }}
  </script>
</body>
</html>'''
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a local HTML dashboard for the Binance 9AM short model.")
    ap.add_argument("--model-dir", default="outputs_9am_rank_model_365_vol100m_test180", help="Model output directory")
    ap.add_argument("--outdir", default="dashboard_9am_short", help="Dashboard output directory")
    ap.add_argument("--top-n", type=int, default=3, help="Maximum short candidates to display")
    ap.add_argument("--short-threshold", type=float, default=0.45, help="Use symbols with prob_up <= this threshold")
    ap.add_argument("--cost-pct", type=float, default=0.50, help="Cost percent used in audit file name")
    ap.add_argument("--min-quote-volume", type=float, default=0.0, help="Optional additional quote-volume filter. 0 means no extra filter.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    min_quote = args.min_quote_volume if args.min_quote_volume and args.min_quote_volume > 0 else None
    latest_rows, latest_meta = load_latest_predictions(model_dir, min_quote)
    metrics, metrics_status = load_model_metrics(model_dir)
    audit_data, audit_status, short_audit, long_audit = load_audit(model_dir, args.top_n, args.cost_pct, args.short_threshold)
    daily_rows, daily_path = load_daily_returns(short_audit)

    html = build_html(
        model_dir=model_dir,
        outdir=outdir,
        latest_rows=latest_rows,
        latest_meta=latest_meta,
        metrics=metrics,
        metrics_status=metrics_status,
        audit_data=audit_data,
        audit_status=audit_status,
        short_audit=short_audit,
        long_audit=long_audit,
        daily_rows=daily_rows,
        top_n=args.top_n,
        short_threshold=args.short_threshold,
        cost_pct=args.cost_pct,
        min_quote_volume=min_quote,
    )
    out = outdir / "index.html"
    out.write_text(html, encoding="utf-8")

    # Also write a tiny metadata file so the user can inspect how it was built.
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(model_dir),
        "outdir": str(outdir),
        "top_n": args.top_n,
        "short_threshold": args.short_threshold,
        "cost_pct": args.cost_pct,
        "min_quote_volume": min_quote,
        "latest_predictions": latest_meta,
        "model_metrics": metrics_status,
        "audit": audit_status,
        "daily_return_file": str(daily_path) if daily_path else None,
    }
    (outdir / "dashboard_build_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Dashboard saved: {out}")
    print(f"Open this file in your browser: {out.resolve()}")


if __name__ == "__main__":
    main()
