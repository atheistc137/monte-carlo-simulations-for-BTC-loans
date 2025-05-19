#!/usr/bin/env python
"""
BTC 80 %-Strike Put Back‑Test — Daily Resolution (v2.3.2)
========================================================

* Monthly‑normalised option cost (30‑day basis).
* `--include-pnl` flag toggles net vs. gross monthly cost.
* `--lookback-years` accepts floats (e.g. 0.1 ≈ 36 days) for quick tests.
* Date rule: **1‑20** ⇒ hedge with current‑month EoM; **21‑EoM** ⇒ next‑month EoM.
"""
from __future__ import annotations

import argparse
import calendar
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_URL = "https://www.deribit.com/api/v2/public"
HEADERS = {"User-Agent": "Deribit-80pct-Put-Backtest/2.3.2"}
ENTRY_HOUR_UTC = 8
STRIKE_STEP = 2_000
RATE_LIMIT_SEC = 0.05
DEFAULT_LOOKBACK_YEARS = 2
CSV_OUT = Path("btc_80pct_puts_raw.csv")
PLOT_OUT = Path("monthly_cost_pct_timeseries.png")

# ---------------------------------------------------------------------------
# Deribit helpers
# ---------------------------------------------------------------------------

def api_get(method: str, params: Dict) -> Optional[Dict]:
    try:
        r = requests.get(f"{BASE_URL}/{method}", params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.HTTPError:
        return None
    payload = r.json()
    if payload.get("error"):
        return None
    time.sleep(RATE_LIMIT_SEC)
    return payload["result"]


def candle_close(instr: str, dt: datetime, *, resolution_min: int = 60) -> Optional[float]:
    start = int((dt - timedelta(minutes=resolution_min)).timestamp() * 1_000)
    end = int(dt.timestamp() * 1_000)
    res = api_get(
        "get_tradingview_chart_data",
        {
            "instrument_name": instr,
            "start_timestamp": start,
            "end_timestamp": end,
            "resolution": str(resolution_min),
        },
    )
    if res is None or res.get("status") != "ok" or not res.get("close"):
        return None
    return res["close"][-1]

# ---------------------------------------------------------------------------
# Instrument helpers
# ---------------------------------------------------------------------------
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

def fmt_instrument(expiry: datetime, strike: int) -> str:
    return f"BTC-{expiry.strftime('%d').upper()}{MONTH_ABBR[expiry.month-1]}{expiry.strftime('%y')}-{strike}-P"


def month_end_expiry(d: date) -> datetime:
    last_day = calendar.monthrange(d.year, d.month)[1]
    cur = date(d.year, d.month, last_day)
    while cur.weekday() not in (3, 4):  # Thu/Fri
        cur -= timedelta(days=1)
    return datetime(cur.year, cur.month, cur.day, 8, tzinfo=timezone.utc)


def target_expiry(entry: datetime) -> datetime:
    if entry.day <= 20:
        return month_end_expiry(entry.date())
    nxt = (entry + relativedelta(months=1)).date().replace(day=1)
    return month_end_expiry(nxt)


def nearest_strike(val: float) -> int:
    return int(round(val / STRIKE_STEP) * STRIKE_STEP)

# ---------------------------------------------------------------------------
# BTC price helpers
# ---------------------------------------------------------------------------

def load_btc_prices(csv: Path) -> pd.Series:
    df = pd.read_csv(csv)
    df["openTime"] = pd.to_datetime(df["openTime"], utc=True)
    return df.set_index("openTime")["close"].sort_index()


def btc_close(series: pd.Series, dt: datetime) -> Optional[float]:
    try:
        return float(series.loc[dt])
    except KeyError:
        return None

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def calc_start(today: datetime, yrs: float) -> date:
    if math.isclose(yrs % 1, 0, abs_tol=1e-6):
        return (today - relativedelta(years=int(yrs))).date()
    return (today - timedelta(days=yrs * 365)).date()

# ---------------------------------------------------------------------------
# DATA PULL
# ---------------------------------------------------------------------------

def pull_data(csv_path: Path, lookback_years: float) -> None:
    price_series = load_btc_prices(csv_path)
    today = datetime.now(timezone.utc).replace(hour=ENTRY_HOUR_UTC, minute=0, second=0, microsecond=0)
    start_date = calc_start(today, lookback_years)
    days = pd.date_range(start_date, today.date(), freq="D", tz=timezone.utc)

    rows: List[Dict] = []
    for ts in tqdm(days, desc="Days"):
        entry_dt = ts.to_pydatetime().replace(hour=ENTRY_HOUR_UTC)
        spot = btc_close(price_series, entry_dt)
        if spot is None:
            continue
        exp_dt = target_expiry(entry_dt)
        if exp_dt <= entry_dt + timedelta(days=7):
            exp_dt += relativedelta(weeks=1)
        dte = (exp_dt.date() - entry_dt.date()).days
        scaler = 30 / dte
        base = nearest_strike(0.8 * spot)
        for k in [base, base - STRIKE_STEP, base + STRIKE_STEP]:
            if k <= 0:
                continue
            name = fmt_instrument(exp_dt, k)
            prem = candle_close(name, entry_dt)
            if prem is None:
                continue
            val_exp = candle_close(name, exp_dt) or 0.0
            rows.append({
                "entry_date": entry_dt.date().isoformat(),
                "instrument": name,
                "strike": k,
                "expiry": exp_dt.date().isoformat(),
                "spot_entry": spot,
                "premium": prem,
                "option_expiry_price": val_exp,
                "days_to_expiry": dte,
                "gross_monthly_pct": prem * 100 * scaler,
                "net_monthly_pct": (prem - val_exp) * 100 * scaler,
            })
            break

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"✔ Saved {len(rows)} rows to {CSV_OUT}")

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_cost(csv_path: Path, *, include_pnl: bool, all_years: bool) -> None:
    df = pd.read_csv(csv_path, parse_dates=["entry_date"])
    if df.empty:
        sys.exit("CSV is empty — nothing to plot.")
    col = "net_monthly_pct" if include_pnl else "gross_monthly_pct"
    if col not in df.columns:
        sys.exit(f"Missing {col} column.")
    if not all_years:
        cutoff = df["entry_date"].max() - pd.DateOffset(years=2)
        df = df[df["entry_date"] >= cutoff]
        if df.empty:
            sys.exit("No recent data to plot.")
    avg = df[col].mean()
    fig, ax = plt.subplots()
    label = "Net monthly cost incl. P&L (% of BTC)" if include_pnl else "Gross monthly cost (% of BTC)"
    ax.plot(df["entry_date"], df[col], label=label)
    ax.axhline(avg, linestyle="--", label=f"Average {avg:.2f}%")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly cost (% of BTC)")
    ax.set_title(f"Hedging through options — {'net' if include_pnl else 'gross'} monthly cost")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    print(f"Plot saved → {PLOT_OUT}")
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="BTC 80% put back‑test (daily, monthly‑scaled cost)")
    p.add_argument("--btc-csv", required=True, type=Path, help="Hourly BTC close‑price CSV")
    p.add_argument("--plot", action="store_true", help="Plot after pulling (or plot existing CSV with --skip-pull)")
    p.add_argument("--skip-pull", action="store_true", help="Skip API pull and use existing CSV")
    p.add_argument("--all-years", action="store_true", help="Plot full history instead of last 2 years")
    p.add_argument("--include-pnl", action="store_true", help="Plot net cost (premium minus expiry value)")
    p.add_argument("--lookback-years", type=float, default=DEFAULT_LOOKBACK_YEARS, help="History window in years (float allowed)")
    args = p.parse_args()

    if not args.btc_csv.exists():
        sys.stderr.write(f"CSV not found: {args.btc_csv}\n")
        sys.exit(1)

    if not args.skip_pull or not CSV_OUT.exists():
        pull_data(args.btc_csv, lookback_years=args.lookback_years)

    if args.plot:
        plot_cost(CSV_OUT, include_pnl=args.include_pnl, all_years=args.all_years)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
