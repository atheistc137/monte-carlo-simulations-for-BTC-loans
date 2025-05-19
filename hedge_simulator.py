#!/usr/bin/env python3
"""
hedge_simulator.py  –  principal-only hedge demo
================================================

One Monte-Carlo path (price + funding) → hourly hedge cash-flow
for a 5-year BTC-backed loan.

Key rules
---------
• Loan principal  = 0.8 × P0  (USD)         – constant, interest-only
• Borrower equity = 0.2 × P0  (USD)         – first-loss cushion
• Hedge size      = USD principal / spot    – but never > 1 BTC
                  = min( 0.8*P0 / P_t , 1.0 )
• Rebalance once per day at 00:00 UTC.
• Cash wallet tracks variation margin, funding P&L, +12 % borrower interest.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from loan_simulator import load_price_funding, bootstrap_rf  # reuse helpers

# ---------- PARAMETERS -----------------
CSV_PATH        = Path("data/BTCUSDT_1h_with_funding.csv")
YEARS_BACK_DATA = 5
SIM_YEARS       = 5
BLOCK           = 24                 # bootstrap block (hours)
APR_INTEREST    = 0.12
REBALANCE_H     = 24
SEED            = 7

np.random.seed(SEED)

# ---------- 1.  Fetch & sample one path ------------
df_rf = load_price_funding(CSV_PATH, years_back=YEARS_BACK_DATA)

hours = SIM_YEARS * 365 * 24
path_rf = bootstrap_rf(df_rf, hours, block=BLOCK)   # [return, funding]

# price path
price = np.empty(hours)
price[0] = 100_000.0                              # set P0 for clarity
for i in range(1, hours):
    price[i] = price[i-1] * (1 + path_rf[i, 0])

returns = path_rf[:, 0]
funding = path_rf[:, 1]

# ---------- 2.  Hedge simulation -------------------
P0 = price[0]
loan_principal = 0.8 * P0          # USD, constant
deposit_usd    = 0.2 * P0          # not used yet

# initial hedge size: USD principal / P0 = 0.8 BTC
Q = loan_principal / P0
cash = 0.0
cash_trace = np.empty(hours)

for t in tqdm(range(hours), desc="hedge"):
    # variation margin on short
    if t > 0:
        cash += -Q * (price[t] - price[t-1])

    # hourly funding (short receives if funding>0)
    cash += Q * price[t] * funding[t]

    # borrower pays interest (interest-only loan)
    cash += loan_principal * APR_INTEREST / 8760

    # daily rebalance
    if t % REBALANCE_H == 0 and t != 0:
        Q_target = min(loan_principal / price[t], 1.0)  # cap at 1 BTC
        delta_Q  = Q_target - Q
        cash    -= delta_Q * price[t]                   # trade cost
        Q        = Q_target

    cash_trace[t] = cash

# ---------- 3.  Plot result -------------------------
days_axis = np.arange(hours) / 24
plt.figure(figsize=(10, 5))
plt.plot(days_axis, cash_trace, label="Cash buffer (USD)")
plt.axhline(0, color="gray", linestyle=":")
plt.title("Principal-only hedge cash buffer • single Monte-Carlo path")
plt.xlabel("Time (days)")
plt.ylabel("Cumulative cash (USD)")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nPeak cash draw-down : {cash_trace.min():,.2f} USD")
print(f"Final cash balance  : {cash_trace[-1]:,.2f} USD")
