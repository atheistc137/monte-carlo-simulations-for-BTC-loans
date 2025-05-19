#!/usr/bin/env python
"""
BTC 80 %-Strike Put Back‑Test — CSV‑Driven (v1.3.4)
=================================================

Changes in **v1.3.4**
--------------------
* Previous commit was truncated — the CLI parser and `if __name__ == "__main__"` block
  are now fully restored.
* Keeps the requested cosmetics:
  * Title → **“Hedging through options”**
  * Line‑series label → **“Monthly cost of hedge as a % of current BTC price”**
* Default plot shows the **last 2 years**; use `--all-years` for the whole set.

Typical usage
-------------
```bash
# Pull (if needed) + plot last 2 years
python btc_puts_80pct_csv.py --btc-csv data/BTCUSDT_1h_with_funding.csv --plot

# Plot entire history without re‑pulling
python btc_puts_80pct_csv.py --btc-csv data/BTCUSDT_1h_with_funding.csv \
    --plot --skip-pull --all-years
```
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_URL = "https://www.deribit.com/api/v2/public"
HEADERS = {"User-Agent": "Deribit-80pct-Put-Backtest/1.3.4"}
LOOKBACK_YEARS = 5
ENTRY_HOUR_UTC = 8               # 08:00 UTC on the first of each month
OPTION_EXPIRY_MIN_DAYS = 21      # 3 weeks
OPTION_EXPIRY_MAX_DAYS = 45      # 6 weeks
STRIKE_STEP = 2_000              # Deribit strike spacing
RATE_LIMIT_SEC = 0.22            # polite rate‑limit

CSV_OUT = Path("btc_80pct_puts_raw.csv")
PLOT_OUT = Path("premium_pct_timeseries.png")

# ---------------------------------------------------------------------------
# Deribit — minimal REST helper
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


def candle_close(instrument: str, dt: datetime, *, resolution_min: int = 60) -> Optional[float]:
    start = int((dt - timedelta(minutes=resolution_min)).timestamp() * 1_000)
    end = int(dt.timestamp() * 1_000)
    res = api_get("get_tradingview_chart_data", {
        "instrument_name": instrument,
        "start_timestamp": start,
        "end_timestamp": end,
        "resolution": str(resolution_min),
    })
    if res is None or res.get("status") != "ok" or not res.get("close"):
        return None
    return res["close"][-1]

# ---------------------------------------------------------------------------
# Instrument helpers
# ---------------------------------------------------------------------------
MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def fmt_instrument(expiry: datetime, strike: int) -> str:
    return f"BTC-{expiry.strftime('%d').upper()}{MONTH_ABBR[expiry.month-1]}{expiry.strftime('%y')}-{strike}-P"


def candidate_expiries(entry_dt: datetime) -> List[datetime]:
    """Return Thursday + Friday expiries 21–45 days after *entry_dt*."""
    start = entry_dt + timedelta(days=OPTION_EXPIRY_MIN_DAYS)
    end = entry_dt + timedelta(days=OPTION_EXPIRY_MAX_DAYS)
    fridays = pd.date_range(start, end, freq="W-FRI", tz=timezone.utc)
    thursdays = pd.date_range(start, end, freq="W-THU", tz=timezone.utc)
    return [d.to_pydatetime() for d in sorted(set(fridays).union(thursdays))]


def nearest_strike(value: float) -> int:
    return int(round(value / STRIKE_STEP) * STRIKE_STEP)

# ---------------------------------------------------------------------------
# BTC spot prices — local CSV
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
# DATA‑PULL ROUTINE
# ---------------------------------------------------------------------------

def pull_data(btc_csv: Path) -> None:
    btc_series = load_btc_prices(btc_csv)
    today = datetime.now(timezone.utc)
    first_month = (today - relativedelta(years=LOOKBACK_YEARS)).replace(day=1, hour=ENTRY_HOUR_UTC, minute=0, second=0, microsecond=0)
    months = [first_month + relativedelta(months=i) for i in range(LOOKBACK_YEARS * 12)]

    rows: List[Dict] = []
    for entry_dt in tqdm(months, desc="Months"):
        spot = btc_close(btc_series, entry_dt)
        if spot is None:
            continue

        base_strike = nearest_strike(0.8 * spot)
        strikes_to_try = [base_strike, base_strike - STRIKE_STEP, base_strike + STRIKE_STEP]

        found = False
        for strike in [s for s in strikes_to_try if s > 0]:
            for expiry_dt in candidate_expiries(entry_dt):
                name = fmt_instrument(expiry_dt, strike)
                premium = candle_close(name, entry_dt)
                if premium is not None:
                    underlying_exp = candle_close("BTC-PERPETUAL", expiry_dt)
                    rows.append({
                        "entry_date": entry_dt.date().isoformat(),
                        "instrument": name,
                        "strike": strike,
                        "expiry": expiry_dt.date().isoformat(),
                        "spot_entry": spot,
                        "premium": premium,
                        "premium_pct": premium * 100,        # premiums quoted in BTC
                        "underlying_exp": underlying_exp,
                    })
                    found = True
                    break
            if found:
                break

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"✔ Saved {len(rows)} rows → {CSV_OUT}")

# ---------------------------------------------------------------------------
# PLOT ROUTINE
# ---------------------------------------------------------------------------

def plot_premium_pct(csv_path: Path, *, all_years: bool = False) -> None:
    df = pd.read_csv(csv_path, parse_dates=["entry_date"])
    if df.empty:
        sys.exit("CSV is empty — nothing to plot.")

    if "premium_pct" not in df.columns:
        df["premium_pct"] = df["premium"] * 100

    if not all_years:
        cutoff = df["entry_date"].max() - pd.DateOffset(years=2)
        df = df[df["entry_date"] >= cutoff]
        if df.empty:
            sys.exit("No data in the last two years to plot.")

    avg_pct = df["premium_pct"].mean()

    fig, ax = plt.subplots()
    ax.plot(
        df["entry_date"],
        df["premium_pct"],
        label="Monthly cost of hedge as a % of current BTC price",
    )
    ax.axhline(avg_pct, linestyle="--", label=f"Average {avg_pct:.2f}%")
    ax.set_xlabel("Month")
    ax.set_ylabel("Premium (% of BTC)")
    ax.set_title("Hedging through options")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()

    fig.savefig(PLOT_OUT, dpi=150)
    print(f"Plot saved → {PLOT_OUT}")
    plt.show()

# ---------------------------------------------------------------------------
# CLI ENTRY‑POINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BTC 80 % put back‑test (pull + plot)")
    parser.add_argument("--btc-csv", required=True, type=Path, help="Hourly BTC close‑price CSV")
    parser.add_argument("--plot", action="store_true", help="Produce plot after pulling (or just plot with --skip-pull)")
    parser.add_argument("--skip-pull", action="store_true", help="Skip Deribit API pull; use existing CSV")
    parser.add_argument("--all-years", action="store_true", help="Plot full history instead of last 2 years")
    args = parser.parse_args()

    if not args.btc_csv.exists():
        sys.exit(f"CSV not found: {args.btc_csv}")

    if not args.skip_pull or not CSV_OUT.exists():
        pull_data(args.btc_csv)

    if args.plot:
        plot_premium_pct(CSV_OUT, all_years=args.all_years)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
