# ================================================================
# NIV Engine v6
# ================================================================

import os, glob, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
import yaml
import argparse


# ---------------- CONFIG ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/config.yaml")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

p = cfg.get("params", {})
ALPHA = p.get("alpha1", 0.5)
BETA = p.get("alpha2", 0.3)
GAMMA = p.get("alpha3", -0.2)
LAMBDA = p.get("lambda", 0.05)
ETA = p.get("eta", 1.0)

print(f"Parameters: α1={ALPHA}, α2={BETA}, α3={GAMMA}, λ={LAMBDA}, η={ETA}")

WINDOW_IMPULSE = 12
WINDOW_DRAG = 24
WINDOW_LSI_SHORT = 6
WINDOW_LSI_LONG = 24

# ---------------- FRED API ----------------
FRED_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_KEY)

OUTDIR = "visuals/v6"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- FETCH DATA ----------------
print("Fetching FRED series...")
series = {
    "GDP": "GDPMC1",
    "INVEST": "GPDIC1",
    "M2": "M2SL",
    "FEDFUNDS": "FEDFUNDS",
    "RND": "DSERRD3A086NBEA",
    "EDU": "W211RC1A027NBEA",
    "TCU": "TCU",
    "DGS10": "DGS10",
    "TB3MS": "TB3MS",
    "TDSP": "TDSP"
}

frames = []
for name, sid in series.items():
    try:
        s = fred.get_series(sid)
        df = pd.DataFrame({"DATE": s.index, name: s.values})
        frames.append(df)
        print(f"{name} loaded: {len(df)} rows")
    except Exception as e:
        print(f"Skipped {name}: {e}")

df = frames[0]
for frame in frames[1:]:
    df = pd.merge(df, frame, on="DATE", how="outer")

df = df.sort_values("DATE").ffill()

# ---------------- CORE EQUATIONS ----------------
print("Computing components...")

# Δ terms
df["ΔG_t"] = df["INVEST"].pct_change().clip(-0.5, 0.5)
df["ΔA_t"] = df["M2"].pct_change(12).rolling(3, min_periods=1).mean().clip(-0.5, 0.5)
df["Δr_t"] = df["FEDFUNDS"].diff() / 100.0

# Exponential kernel
kernel = np.exp(-LAMBDA * np.arange(WINDOW_IMPULSE))
kernel /= kernel.sum()

def discounted(series):
    arr = series.fillna(0).to_numpy()
    out = np.convolve(arr, kernel, mode="same")
    return pd.Series(out, index=series.index)

# === tanh(u_t) — FROM YOUR PDF ===
raw_impulse = ALPHA*df["ΔG_t"] + BETA*df["ΔA_t"] + GAMMA*df["Δr_t"]
df["u_t"] = np.tanh(discounted(raw_impulse))

# Regeneration share
df["P_t"] = (df["INVEST"].fillna(0) + df["RND"].fillna(0) + df["EDU"].fillna(0)) / df["GDP"].replace(0, np.nan)

# Idle capacity
df["X_t"] = (1 - df["TCU"]/100).clip(0, 1)

# Friction
df["term_spread"] = (df["DGS10"] - df["TB3MS"]).clip(lower=0)
df["debt_share"] = (df["TDSP"] / df["GDP"]).fillna(0)
df["F_t"] = df["term_spread"] + df["debt_share"]

# NIV core
df["NIV_t"] = (df["u_t"] * (df["P_t"]**2)) / (df["X_t"] + df["F_t"])**ETA

# ---------------- NORMALIZER ----------------
print("Running normalizer...")
df["NIV_t"] = np.sign(df["NIV_t"]) * np.log1p(np.abs(df["NIV_t"]))
rolling_std = df["NIV_t"].rolling(120, min_periods=30).std().clip(lower=0.05)
df["NIV_t"] = (df["NIV_t"] / rolling_std).clip(-10, 10) * 5

# === INDEX ENDS AT 100 ===
latest_niv = df["NIV_t"].iloc[-1]
if latest_niv == 0:
    latest_niv = 1e-6  # Prevent division by zero
df["NIV_Index"] = 100 * df["NIV_t"] / latest_niv

# Diagnostics
df["Impulse"] = df["NIV_t"].diff(WINDOW_IMPULSE)
df["Drag"] = df["NIV_t"].rolling(WINDOW_DRAG).std()
df["LSI"] = df["NIV_t"].rolling(WINDOW_LSI_SHORT).std() / df["NIV_t"].rolling(WINDOW_LSI_LONG).std()

# ---------------- EXPORT ----------------
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTDIR, f"niv_processed_v6_{ts}.csv")
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")

# ---------------- VISUALS ----------------
def plot_series(y, title, fname, color="#00e7ff", log=False):
    plt.figure(figsize=(14,7))
    plt.plot(df["DATE"], df[y], color=color, lw=2.2)
    if log: plt.yscale("log")
    plt.title(title, fontsize=16, fontweight='bold', color='#e0e0e0')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.grid(True, alpha=0.3, color='#334155')
    plt.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=300, facecolor='#0f172a')
    plt.close()
    print(f"{fname}")

# ---------------- NORMALIZER + INDEX ENDS AT 100 ----------------
print("Running normalizer...")
# Log compression
df["NIV_t_raw"] = df["NIV_t"].copy()  # Save raw before normalization
df["NIV_t"] = np.sign(df["NIV_t"]) * np.log1p(np.abs(df["NIV_t"]))

# Rolling std normalization
rolling_std = df["NIV_t"].rolling(120, min_periods=30).std().clip(lower=0.05)
df["NIV_t"] = (df["NIV_t"] / rolling_std).clip(-10, 10) * 5

# === INDEX ENDS AT 100 (USING RAW LAST VALUE) ===
latest_raw = df["NIV_t_raw"].iloc[-1]
if abs(latest_raw) < 1e-12:
    latest_raw = 1e-6
df["NIV_Index"] = 100 * df["NIV_t_raw"] / latest_raw  # ← ENDS AT 100

# Diagnostics (use normalized NIV_t)
df["Impulse"] = df["NIV_t"].diff(WINDOW_IMPULSE)
df["Drag"] = df["NIV_t"].rolling(WINDOW_DRAG).std()
df["LSI"] = df["NIV_t"].rolling(WINDOW_LSI_SHORT).std() / df["NIV_t"].rolling(WINDOW_LSI_LONG).std()

# ---------------- VISUALS ----------------
def plot_series(y, title, fname, color="#00e7ff", log=False):
    plt.figure(figsize=(14,7))
    plt.plot(df["DATE"], df[y], color=color, lw=2.2)
    if log: plt.yscale("log")
    plt.title(title, fontsize=16, fontweight='bold', color='#e0e0e0')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.grid(True, alpha=0.3, color='#334155')
    plt.tight_layout()
    path = os.path.join(OUTDIR, fname)
    plt.savefig(path, dpi=300, facecolor='#0f172a')
    plt.close()
    print(f"{fname}")

# === INDEX ENDS AT 100 ===
plot_series("NIV_Index", "National Impact Velocity Index (Ends at 100)", f"niv_gdpnorm_{ts}.png")
plot_series("u_t", "Activation Intensity u_t = tanh(...)", f"u_t_{ts}.png", color="#7c3aed")
plot_series("Impulse", "Throughput Impulse", f"niv_impulse_{ts}.png", color="#10b981")
plot_series("Drag", "Structural Drag", f"niv_drag_{ts}.png", color="#f59e0b")
plot_series("LSI", "Liquidity Stress Intensity", f"niv_lsi_{ts}.png", color="#ef4444")

print("NIV Engine v6 complete")
