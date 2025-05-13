#!/usr/bin/env python3
"""
loan_simulator.py
=================
Monte‑Carlo liquidation probability for a BTC‑backed loan.

Parameters are loaded from config.json (see earlier version).  
This revision gracefully handles **hedge_ratio = 1.0** (“perfect hedge”)
by defining liquidation probability ≡ 0 % and avoiding divide‑by‑zero.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import binance_loader

# -----------------------------  CONFIG  --------------------------------
DEFAULT_CFG = {
    "years":            5,
    "deposit_fraction": 0.20,
    "hedge_ratio":      0.0
}


def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"[config] {path} not found – using defaults {DEFAULT_CFG}")
        return DEFAULT_CFG.copy()
    cfg = DEFAULT_CFG.copy()
    cfg.update(json.loads(path.read_text()))
    return cfg


# -------------------------  MONTE‑CARLO CORE  --------------------------
def _block_bootstrap(rets: np.ndarray, steps: int, block: int = 24) -> np.ndarray:
    out = np.empty(steps)
    i, n = 0, len(rets)
    while i < steps:
        start = np.random.randint(0, n - block)
        chunk = rets[start:start + block]
        take  = min(block, steps - i)
        out[i:i + take] = chunk[:take]
        i += take
    return out


def _simulate_path(rets: np.ndarray, years: int,
                   deposit_frac: float, hedge_ratio: float) -> tuple[bool, float]:
    """
    Returns
    -------
    liquidated : bool
    min_ratio  : float   Minimum price ÷ initial price
    """
    # ------- Perfect hedge: no price risk, never liquidates --------------
    if hedge_ratio >= 1.0 - 1e-12:          # tolerate tiny FP wiggle
        return False, 1.0                   # price path irrelevant

    hours = years * 365 * 24
    init_p = 1.0
    liq_frac = deposit_frac / (1 - hedge_ratio)
    liq_ratio = 1 - liq_frac

    path = init_p * np.cumprod(1 + _block_bootstrap(rets, hours))
    min_ratio = path.min() / init_p
    return min_ratio <= liq_ratio, min_ratio


def mc_run(rets: np.ndarray, sims: int,
           years: int, deposit_frac: float, hedge_ratio: float) -> tuple[float, list[float]]:
    """
    Returns
    -------
    prob : float               Liquidation probability
    mins : list[float]         Min‑ratio for each path (empty if hedge=1)
    """
    # Fast‑path perfect hedge: probability = 0, no mins needed
    if hedge_ratio >= 1.0 - 1e-12:
        return 0.0, [1.0] * sims

    mins, hits = [], 0
    for _ in tqdm(range(sims), desc="Simulating"):
        liq, m = _simulate_path(rets, years, deposit_frac, hedge_ratio)
        mins.append(m)
        if liq:
            hits += 1
    return hits / sims, mins


# ---------------------------  CLI & MAIN  ------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BTC loan liquidation Monte‑Carlo")
    p.add_argument("--symbol", default="BTCUSDT", help="Binance symbol")
    p.add_argument("--csv", default="data/BTCUSDT_1h.csv",
                   help="Hourly CSV cache path")
    p.add_argument("--start", default="2017-01-01",
                   help="Fetch start date if CSV missing")
    p.add_argument("--sims", type=int, default=20000, help="Paths")
    p.add_argument("--config", default="config.json",
                   help="JSON with years, deposit_fraction, hedge_ratio")
    p.add_argument("--plot", action="store_true",
                   help="Show histogram of min price ratios")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    # --- data -----------------------------------------------------------
    csv = Path(args.csv)
    if not csv.exists():
        binance_loader.download_hourly(args.symbol, csv, args.start)

    df = pd.read_csv(csv, index_col=0, parse_dates=[0])
    prices = df["close"].astype(float).values
    returns = np.diff(prices) / prices[:-1]

    prob, mins = mc_run(returns,
                        sims=args.sims,
                        years=cfg["years"],
                        deposit_frac=cfg["deposit_fraction"],
                        hedge_ratio=cfg["hedge_ratio"])

    print(f"\nLiquidation probability over {cfg['years']} yrs "
          f"(deposit={cfg['deposit_fraction']:.2f}, "
          f"hedge={cfg['hedge_ratio']:.2f}): {prob*100:.2f}%")

    # --- optional plot --------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt

        if cfg["hedge_ratio"] >= 1.0 - 1e-12:
            print("[plot] Perfect hedge – no liquidation threshold to show.")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "Perfect hedge\n(no liquidation)",
                     ha="center", va="center", fontsize=14)
            plt.axis("off")
            plt.show()
            return

        liq_ratio = 1 - cfg["deposit_fraction"] / (1 - cfg["hedge_ratio"])
        plt.figure(figsize=(8, 4.8))
        bins = np.linspace(0, 1, 120)
        plt.hist(mins, bins=bins, alpha=0.75, edgecolor="black")
        plt.axvline(liq_ratio, color="red", linestyle="--",
                    label=f"Liquidation threshold ({liq_ratio:.2f})")
        plt.title("Minimum price ratio over simulation horizon")
        plt.xlabel("min(price)/initial price")
        plt.ylabel("Number of paths")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
