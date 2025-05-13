#!/usr/bin/env python3
"""
binance_loader.py
=================
Utility functions for downloading & caching hourly klines from Binance.

Key entry point
---------------
download_hourly(symbol="BTCUSDT",
                csv_path="data/BTCUSDT_1h.csv",
                start="2017-01-01",
                end=None,
                force=False)  -> pandas.DataFrame
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import pandas as pd

# -------------  CONFIG  -------------------------------------------------
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000                  # Binance API row cap
# -----------------------------------------------------------------------


def _to_ms(dt: datetime) -> int:
    """datetime -> milliseconds since epoch (UTC)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fetch_hourly(symbol: str,
                  start_dt: datetime,
                  end_dt: datetime) -> pd.DataFrame:
    """Low‑level fetch that talks to Binance REST in a paging loop."""
    rows: list[list] = []
    curr = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)

    while curr < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": curr,
            "endTime":   end_ms,
            "limit":     MAX_LIMIT
        }
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        curr = chunk[-1][6] + 1          # closeTime + 1 ms
        time.sleep(0.1)                  # polite pacing

    df = pd.DataFrame(rows, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","quoteVol","trades","takerBaseVol",
        "takerQuoteVol","ignore"
    ])
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df["close"]    = df["close"].astype(float)
    return df.set_index("openTime")[["close"]].sort_index()


def download_hourly(symbol: str = "BTCUSDT",
                    csv_path: str | Path = "data/BTCUSDT_1h.csv",
                    start: str | datetime = "2017-01-01",
                    end: str | datetime | None = None,
                    force: bool = False):
    """
    Fetch hourly data (if not cached) and return DataFrame.
    * If the CSV already exists and force=False, just load it.
    * Otherwise fetch from Binance, save to CSV, and return DF.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not force:
        print(f"[binance_loader] Using cached data: {csv_path}")
        return pd.read_csv(csv_path, parse_dates=[0], index_col=0)

    start_dt = (datetime.fromisoformat(start)
                if isinstance(start, str) else start)
    end_dt   = (datetime.utcnow()
                if end is None else
                (datetime.fromisoformat(end) if isinstance(end, str) else end))

    print(f"[binance_loader] Downloading {symbol} 1‑h klines "
          f"{start_dt.date()} → {end_dt.date()} …")
    df = _fetch_hourly(symbol, start_dt, end_dt)
    df.to_csv(csv_path)
    print(f"[binance_loader] Saved {len(df):,} rows to {csv_path}")
    return df
