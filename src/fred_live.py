import pandas as pd
import numpy as np
import os
from datetime import datetime

# === Paths ===
INPUT = "fred_live.csv"
OUT = f"niv_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# === Load Data ===
df = pd.read_csv(INPUT, parse_dates=["DATE"]).set_index("DATE")
print("Columns found:", list(df.columns))

# --- Step 1: Interpolation and Early-Era Backfill ---
for col in ["GPDIC1", "M2SL", "FEDFUNDS", "GDPMC1", "TCU"]:
    if col in df.columns:
        df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

if "FEDFUNDS" in df.columns:
    df["FEDFUNDS"].fillna(1.5, inplace=True)

df["M2SL"] = df["M2SL"].rolling(3, min_periods=1).mean()

# Resample to quarterly
df = df.resample("Q").mean()

# --- Step 2: Derivative Components ---
dGov = df["GPDIC1"].pct_change().clip(-0.5, 0.5)
dM2 = df["M2SL"].pct_change(4).rolling(3, min_periods=1).mean().clip(-0.5, 0.5)

util = df["TCU"] / 100 if "TCU" in df.columns else 0.8
rshort = df["FEDFUNDS"] / 100
rlong = df["DGS10"] / 100 if "DGS10" in df.columns else rshort

# --- Step 3: Construct CVE State Variables ---
# === tanh(u_t) ADDED HERE ===
raw_impulse = 0.8*dGov + 0.6*dM2 - 0.3*(rshort - rlong)
df["u_t"] = np.tanh(raw_impulse.rolling(3).mean())  # ← tanh BOUNDS u_t ∈ [-1,1]

df["X_t"] = (dGov * util).fillna(0)
df["F_t"] = (1 - dM2).clip(0, 2)
df["P_t"] = (dM2 * util).fillna(0)

# --- Step 4: Core CVE Equation (NIV_t) ---
df["NIV_t"] = (df["u_t"] * df["P_t"] * df["X_t"]) / (df["F_t"] + 1e-6)
df["NIV_t"] = df["NIV_t"].replace([np.inf, -np.inf], np.nan).fillna(0)

# --- Step 5: GDP Normalization ---
GDP = df["GDPMC1"].ffill().clip(lower=1e-6)
df["NIV_t"] = df["NIV_t"] / GDP

# --- Step 6: Logarithmic Compression and Rolling Variance Normalization ---
df["NIV_t"] = np.sign(df["NIV_t"]) * np.log1p(np.abs(df["NIV_t"]))
rolling_std = df["NIV_t"].rolling(120, min_periods=30).std().clip(lower=0.05)
df["NIV_t"] = (df["NIV_t"] / rolling_std).clip(-10, 10) * 5

print(f"NIV stabilized: mean={df['NIV_t'].mean():.4f}, std={df['NIV_t'].std():.4f}")

# --- Step 7: Liquidity Stress Intensity + Structural Drag ---
df["LSI"] = (
    df["F_t"].rolling(12, min_periods=1).std()
    / (df["F_t"].rolling(60, min_periods=1).std() + 1e-6)
).fillna(0)

df["Drag"] = df["NIV_t"].rolling(24, min_periods=1).mean().diff().fillna(0)

# --- Step 8: Export ---
df_out = df[["u_t", "X_t", "F_t", "P_t", "NIV_t", "LSI", "Drag"]]
df_out.to_csv(OUT)
print(f"Saved normalized dataset: {OUT} | Shape: {df_out.shape}")
