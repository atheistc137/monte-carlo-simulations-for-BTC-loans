#!/usr/bin/env python3
"""
grid_liquidation.py
===================
Run the BTC‑loan Monte‑Carlo on a full grid of
deposit_fractions × hedge_ratios read from grid_config.json,
then visualise the liquidation probability as a heat‑map.

--------------------------------------------------------------------
Expected JSON (defaults shown – create grid_config.json to override)
--------------------------------------------------------------------
{
  "years":             5,
  "sims":              10000,
  "deposit_fractions": [0.20, 0.30, 0.40, 0.50],
  "hedge_ratios":      [0.0, 0.25, 0.50, 0.75, 1.0]
}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm

import binance_loader
import loan_simulator                       # re‑use its Monte‑Carlo engine


# --------------  DEFAULT GRID CONFIG  ---------------------------------
DEFAULT_GRID_CFG = {
    "years":             5,
    "sims":              10000,
    "deposit_fractions": [0.20, 0.30, 0.40, 0.50],
    "hedge_ratios":      [0.0,  0.25,  0.50, 0.75, 1.0]
}


def load_grid_cfg(path: Path) -> dict:
    if not path.exists():
        print(f"[grid‑config] {path} not found – using defaults {DEFAULT_GRID_CFG}")
        return DEFAULT_GRID_CFG.copy()
    with path.open() as f:
        user_cfg = json.load(f)
    cfg = DEFAULT_GRID_CFG.copy()
    cfg.update(user_cfg)                         # merge user overrides
    # sort lists to ensure monotone axes
    cfg["deposit_fractions"] = sorted(cfg["deposit_fractions"])
    cfg["hedge_ratios"]      = sorted(cfg["hedge_ratios"])
    print(f"[grid‑config] Loaded {cfg} from {path}")
    return cfg


# --------------  CLI  --------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Grid Monte‑Carlo liquidation explorer")
    p.add_argument("--symbol", default="BTCUSDT", help="Binance pair")
    p.add_argument("--csv", default="data/BTCUSDT_1h.csv",
                   help="Cached hourly CSV path")
    p.add_argument("--start", default="2017-01-01",
                   help="Fetch start date if CSV missing")
    p.add_argument("--config", default="grid_config.json",
                   help="Grid‑parameter JSON file")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_grid_cfg(Path(args.config))

    # 1.  Ensure price data
    csv = Path(args.csv)
    if not csv.exists():
        binance_loader.download_hourly(symbol=args.symbol,
                                       csv_path=csv,
                                       start=args.start,
                                       end=None,
                                       force=False)

    # 2.  Build hourly return vector once
    df = pd.read_csv(csv, index_col=0, parse_dates=[0])
    prices  = df["close"].astype(float).values
    returns = np.diff(prices) / prices[:-1]

    dep_list   = cfg["deposit_fractions"]
    hedge_list = cfg["hedge_ratios"]
    years      = cfg["years"]
    sims       = cfg["sims"]

    Z = np.zeros((len(hedge_list), len(dep_list)))   # rows = hedges

    print("\nRunning Monte‑Carlo grid …")
    for i, h in enumerate(tqdm(hedge_list, desc="Hedge levels")):
        for j, d in enumerate(dep_list):
            prob, _ = loan_simulator.mc_run(returns,
                                            sims=sims,
                                            years=years,
                                            deposit_frac=d,
                                            hedge_ratio=h)
            Z[i, j] = prob

    # 3.  Numeric matrix output
    print("\nLiquidation probability matrix (rows = hedge_ratio, cols = deposit):")
    header = "       " + " ".join(f"{d:6.2f}" for d in dep_list)
    print(header)
    for i, h in enumerate(hedge_list):
        row_vals = " ".join(f"{100*Z[i,j]:6.1f}%" for j in range(len(dep_list)))
        print(f"h={h:4.2f} {row_vals}")

    # 4.  Heat‑map
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto",
                   extent=[min(dep_list), max(dep_list),
                           min(hedge_list), max(hedge_list)])

    # colour bar with % scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Liquidation probability (%)")
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # axis labels & ticks
    ax.set_xlabel("Deposit fraction")
    ax.set_ylabel("Hedge ratio")
    ax.set_title(f"{years}-year liquidation probability • {sims:,} paths")

    ax.set_xticks(dep_list)
    ax.set_yticks(hedge_list)

    # annotate cells
    for i, h in enumerate(hedge_list):
        for j, d in enumerate(dep_list):
            ax.text(d, h, f"{Z[i,j]*100:.0f}%",
                    ha='center', va='center',
                    color='white' if Z[i,j] > 0.5 else 'black',
                    fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
