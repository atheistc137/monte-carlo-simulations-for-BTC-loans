#!/usr/bin/env python3
"""
Merge Binance hourly close prices with hourly funding‑rate data.

  • Reads   ./data/BTCUSDT_1h.csv                (time, close)
  • Fetches realised funding rates from
      2017‑08‑17 04:00:00 UTC  ➜  now           (8‑hour cadence)
  • Forward‑fills to hourly
  • Saves   ./data/BTCUSDT_1h_with_funding.csv   (time, close, fundingRate)
"""

import requests
import pandas as pd
import time as _time
import datetime as dt
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────────────
BASE         = "https://fapi.binance.com"      # USD‑M Futures REST base
SYMBOL       = "BTCUSDT"
START_AT_STR = "2017-08-17 04:00:00+00:00"     # inclusive start of funding pull

# paths relative to the script location
HERE        = Path(__file__).resolve().parent
DATA_DIR    = HERE / "data"
PRICE_PATH  = DATA_DIR / f"{SYMBOL}_1h.csv"
OUT_PATH    = DATA_DIR / f"{SYMBOL}_1h_with_funding.csv"

# ── 1. read existing hourly close‑price file ──────────────────────────────────
price_df = pd.read_csv(
    PRICE_PATH,
    parse_dates=["openTime"],
    index_col="openTime",
    dtype={"close": "float64"},
)
# convert to tz‑naïve UTC for consistent merging
if price_df.index.tz is not None:
    price_df.index = price_df.index.tz_convert(None)

# ── 2. pull funding‑rate history from START_AT ───────────────────────────────
start_ts = pd.Timestamp(START_AT_STR).tz_convert("UTC")
start_window_ms = int(start_ts.timestamp() * 1000)

rows = []
while True:
    resp = requests.get(
        f"{BASE}/fapi/v1/fundingRate",
        params={
            "symbol":    SYMBOL,
            "startTime": start_window_ms,
            "limit":     1000,             # max rows per call
        },
        timeout=15,
    )
    batch = resp.json()
    if not batch:
        break

    rows.extend(batch)
    start_window_ms = batch[-1]["fundingTime"] + 1
    _time.sleep(0.05)                      # respect 500 req / 5 min IP limit

funding_8h = pd.DataFrame(rows)
funding_8h["fundingTime"] = pd.to_datetime(
    funding_8h["fundingTime"], unit="ms", utc=True
)
funding_8h.set_index("fundingTime", inplace=True)
funding_8h["fundingRate"] = funding_8h["fundingRate"].astype(float)

# ── 3. resample to hourly and align with price data ──────────────────────────
funding_1h = funding_8h["fundingRate"].resample("1h").ffill()
funding_1h.index = funding_1h.index.tz_convert(None)    # tz‑naïve UTC

# restrict to the price index span and forward‑fill any leading gaps
funding_aligned = funding_1h.reindex(price_df.index, method="ffill")

# ── 4. merge and save ────────────────────────────────────────────────────────
combined = price_df.assign(fundingRate=funding_aligned)
combined.reset_index().rename(columns={"index": "time"}).to_csv(
    OUT_PATH, index=False
)

print(f"✓ Created {OUT_PATH.relative_to(HERE)}")
