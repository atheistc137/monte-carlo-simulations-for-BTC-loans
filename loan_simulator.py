#!/usr/bin/env python3
"""
loan_simulator.py  •  v2.1  (tz‑safe price + funding bootstrap)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------- #
# 1  Config                                                             #
# --------------------------------------------------------------------- #
DEFAULT_CFG = {
    "years":            5,
    "deposit_fraction": 0.20,
    "hedge_ratio":      0.0
}


def load_cfg(path: Path) -> dict:
    if not path.exists():
        print(f"[config] {path} not found – using defaults {DEFAULT_CFG}")
        return DEFAULT_CFG.copy()
    cfg = DEFAULT_CFG.copy()
    cfg.update(json.loads(path.read_text()))
    return cfg


# --------------------------------------------------------------------- #
# 2  Load price + funding (UTC‑aware)                                   #
# --------------------------------------------------------------------- #
def load_price_funding(csv_path: Path, years_back: int = 5) -> pd.DataFrame:
    """
    Returns DataFrame indexed by **UTC tz‑aware** datetime with columns:
        r – hourly pct return
        f – per‑hour funding rate
    """
    df = pd.read_csv(csv_path, parse_dates=["openTime"])
    df = df.set_index("openTime").sort_index()

    # localise if tz‑naive
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    cutoff = datetime.now(timezone.utc) - timedelta(days=years_back * 365)
    df = df.loc[df.index >= cutoff]

    df["r"] = df["close"].pct_change()
    df["f"] = df["fundingRate"]
    df = df.dropna(subset=["r", "f"])
    return df[["r", "f"]]


# --------------------------------------------------------------------- #
# 3  Bootstrap sampler (unchanged)                                      #
# --------------------------------------------------------------------- #
def bootstrap_rf(df: pd.DataFrame, steps: int, block: int = 24) -> np.ndarray:
    out = np.empty((steps, 2))
    i, n = 0, len(df)
    while i < steps:
        start = np.random.randint(0, n - block)
        take = min(block, steps - i)
        out[i:i+take] = df.iloc[start:start+take].to_numpy()[:take]
        i += take
    return out


# --------------------------------------------------------------------- #
# 4.  Single‑path simulation                                            #
# --------------------------------------------------------------------- #
def simulate_path(rf: np.ndarray,
                  deposit_frac: float,
                  hedge_ratio: float,
                  init_price: float = 1.0) -> tuple[bool, float, float]:
    """
    Parameters
    ----------
    rf           : ndarray (steps, 2) [return, funding]
    deposit_frac : borrower deposit (e.g. 0.20)
    hedge_ratio  : BTC short per 1 BTC collateral

    Returns
    -------
    liquidated : bool
    min_ratio  : float   (min price / init price)
    sum_funding: float   cumulative funding income on *1 BTC short* (for diag)
    """
    # perfect hedge: never liquidates, funding inconsequential for now
    if hedge_ratio >= 1.0 - 1e-12:
        return False, 1.0, 0.0

    price = init_price
    liq_frac = deposit_frac / (1 - hedge_ratio)
    liq_ratio = 1 - liq_frac
    min_ratio = 1.0
    funding_sum = 0.0

    for r, f in rf:
        price *= 1 + r
        min_ratio = min(min_ratio, price / init_price)
        funding_sum += f            # future: multiply by hedge notional

        if price / init_price <= liq_ratio:
            return True, min_ratio, funding_sum
    return False, min_ratio, funding_sum


# --------------------------------------------------------------------- #
# 5.  Monte‑Carlo driver                                                #
# --------------------------------------------------------------------- #
def mc_run(df_rf: pd.DataFrame,
           sims: int,
           years: int,
           deposit_frac: float,
           hedge_ratio: float) -> tuple[float, float]:
    """
    Returns
    -------
    prob_liq   : float
    mean_fund  : float   (average cumulative funding per BTC short)
    """
    if hedge_ratio >= 1.0 - 1e-12:
        print("[mc] Perfect hedge detected – liquidation prob = 0%")
        return 0.0, 0.0

    steps = years * 365 * 24
    liq_hits = 0
    funding_accum = 0.0

    for _ in tqdm(range(sims), desc="Simulating"):
        rf_path = bootstrap_rf(df_rf, steps)
        liq, _, fund_sum = simulate_path(rf_path, deposit_frac, hedge_ratio)
        if liq:
            liq_hits += 1
        funding_accum += fund_sum

    return liq_hits / sims, funding_accum / sims


# --------------------------------------------------------------------- #
# 6.  CLI / main                                                        #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BTC loan Monte‑Carlo (price + funding)")
    p.add_argument("--csv", default="data/BTCUSDT_1h_with_funding.csv",
                   help="CSV with hourly price & funding")
    p.add_argument("--sims", type=int, default=15000, help="MC paths")
    p.add_argument("--config", default="config.json",
                   help="JSON file with years, deposit_fraction, hedge_ratio")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_cfg(Path(args.config))

    df_rf = load_price_funding(Path(args.csv), years_back=5)

    prob, avg_funding = mc_run(df_rf,
                               sims=args.sims,
                               years=cfg["years"],
                               deposit_frac=cfg["deposit_fraction"],
                               hedge_ratio=cfg["hedge_ratio"])

    print(f"\n--- Monte‑Carlo summary (last 5 yrs data) ---")
    print(f"Deposit         : {cfg['deposit_fraction']:.2f}")
    print(f"Hedge ratio     : {cfg['hedge_ratio']:.2f}")
    print(f"Liquidation prob: {prob*100:.2f}%")
    print(f"Avg funding P&L : {avg_funding:.5f}  (per 1 BTC short, "
          f"{cfg['years']}‑yr horizon)\n")


if __name__ == "__main__":
    main()
