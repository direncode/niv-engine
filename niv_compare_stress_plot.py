# niv_compare_stress_fred_oos_max_rigorous.py
# Maximal rigorous OOS comparison with bootstrap significance, PR-AUC, thresholds, plots (saved)

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import yaml

# Config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)["params"]

ALPHA1 = cfg["alpha1"]
ALPHA2 = cfg["alpha2"]
ALPHA3 = cfg["alpha3"]
LAMBDA = cfg["lambda"]
ETA = cfg["eta"]
WINDOW = 12

# Load extended FRED data
df = pd.read_csv("fred_live_extended.csv", parse_dates=["DATE"], index_col="DATE")

# Your 20 stress peaks
stress_peaks = pd.to_datetime([
    "1966-07-01", "1970-01-01", "1974-01-01", "1980-04-01", "1982-01-01",
    "1987-10-01", "1991-01-01", "1998-10-01", "2001-04-01", "2008-10-01",
    "2011-10-01", "2016-01-01", "2018-04-01", "2018-10-01", "2019-10-01",
    "2020-04-01", "2022-04-01", "2023-04-01", "2025-01-01", "2025-04-01"
])

# Stress period ±3 quarters
stress = pd.Series(0, index=df.index)
for p in stress_peaks:
    if p in df.index:
        start = p - pd.DateOffset(months=9)
        end = p + pd.DateOffset(months=9)
        mask = (df.index >= start) & (df.index <= end)
        stress.loc[mask] = 1

# Target: stress in next 6 quarters
horizon = 6
y = stress.shift(-horizon).rolling(horizon).max().fillna(0)

# Kernel
kernel = np.exp(-LAMBDA * np.arange(WINDOW))
kernel /= kernel.sum()

def discounted(s):
    arr = np.concatenate([np.zeros(WINDOW-1), s.fillna(0).values])
    conv = np.convolve(arr, kernel, mode="valid")
    return pd.Series(conv, index=s.index)

# Rolling OOS
window_size = 120
results = []

for end in range(window_size, len(df) - horizon):
    train = df.iloc[end-window_size:end].copy()
    date = df.index[end]
    
    d = train.copy().interpolate(limit_direction="both").ffill().bfill()
    
    # NIV v6
    d["ΔG_t"] = d["GPDIC1"].pct_change().clip(-0.5, 0.5)
    d["ΔA_t"] = d["M2SL"].pct_change(12).rolling(3, min_periods=1).mean().clip(-0.5, 0.5)
    d["Δr_t"] = d["FEDFUNDS"].diff() / 100
    
    impulse = ALPHA1 * d["ΔG_t"] + ALPHA2 * d["ΔA_t"] + ALPHA3 * d["Δr_t"]
    d["u_t"] = discounted(impulse)
    
    d["P_t"] = d["GPDIC1"] / d["GDPC1"]
    d["X_t"] = (1 - d["TCU"]/100).clip(0, 1)
    d["term_spread"] = (d["DGS10"] - d["TB3MS"]).clip(lower=0)
    d["debt_gdp"] = d["GFDEBTN"] / (d["GDPC1"] * 4)
    d["F_t"] = d["term_spread"] + d["debt_gdp"]
    
    denom = (d["X_t"] + d["F_t"]).replace(0, 1e-6) ** ETA
    d["NIV_t"] = (d["u_t"] * (d["P_t"] ** 2)) / denom
    d["NIV_t"] = d["NIV_t"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    niv = d["NIV_t"].iloc[-1]
    
    # Benchmarks
    nfci = d["NFCI"].iloc[-1] if "NFCI" in d.columns else np.nan
    stlfsi = d["STLFSI"].iloc[-1] if "STLFSI" in d.columns else np.nan
    kcfsi = d["KCFSI"].iloc[-1] if "KCFSI" in d.columns else np.nan
    vix = d["VIX"].iloc[-1] if "VIX" in d.columns else np.nan
    
    results.append({
        "date": date,
        "NIV": niv,
        "NFCI": nfci,
        "STLFSI": stlfsi,
        "KCFSI": kcfsi,
        "VIX": vix,
        "stress_ahead": y.loc[date]
    })

oos = pd.DataFrame(results).set_index("date").dropna()

# AUC-ROC (higher = stress for benchmarks, lower NIV = stress)
auc_niv = roc_auc_score(oos["stress_ahead"], -oos["NIV"])
auc_nfci = roc_auc_score(oos["stress_ahead"], oos["NFCI"])
auc_stlfsi = roc_auc_score(oos["stress_ahead"], oos["STLFSI"])
auc_kcfsi = roc_auc_score(oos["stress_ahead"], oos["KCFSI"])
auc_vix = roc_auc_score(oos["stress_ahead"], oos["VIX"])

# Precision-Recall AUC (better for imbalanced)
pr_niv = auc(*precision_recall_curve(oos["stress_ahead"], -oos["NIV"])[1::-1])
pr_nfci = auc(*precision_recall_curve(oos["stress_ahead"], oos["NFCI"])[1::-1])
pr_stlfsi = auc(*precision_recall_curve(oos["stress_ahead"], oos["STLFSI"])[1::-1])
pr_kcfsi = auc(*precision_recall_curve(oos["stress_ahead"], oos["KCFSI"])[1::-1])
pr_vix = auc(*precision_recall_curve(oos["stress_ahead"], oos["VIX"])[1::-1])

# Bootstrap significance (NIV vs each benchmark)
n_boot = 1000
boot_diff_nfci = []
boot_diff_kcfsi = []
boot_diff_vix = []

for _ in range(n_boot):
    sample = oos.sample(frac=1, replace=True)
    if sample["stress_ahead"].nunique() > 1:
        a_niv = roc_auc_score(sample["stress_ahead"], -sample["NIV"])
        a_nfci = roc_auc_score(sample["stress_ahead"], sample["NFCI"])
        a_kcfsi = roc_auc_score(sample["stress_ahead"], sample["KCFSI"])
        a_vix = roc_auc_score(sample["stress_ahead"], sample["VIX"])
        
        boot_diff_nfci.append(a_niv - a_nfci)
        boot_diff_kcfsi.append(a_niv - a_kcfsi)
        boot_diff_vix.append(a_niv - a_vix)

p_vs_nfci = np.mean(np.array(boot_diff_nfci) >= 0)
p_vs_kcfsi = np.mean(np.array(boot_diff_kcfsi) >= 0)
p_vs_vix = np.mean(np.array(boot_diff_vix) >= 0)

# Optimal thresholds and metrics
def get_metrics(probs, y_true, name):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    best_idx = np.argmax(f1)
    best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0
    sig = (probs > best_thresh).astype(int) if "NIV" not in name else (probs < best_thresh).astype(int)
    
    tp = ((sig==1) & (y_true==1)).sum()
    fp = ((sig==1) & (y_true==0)).sum()
    fn = ((sig==0) & (y_true==1)).sum()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return recall, precision_val

recall_niv, precision_niv = get_metrics(-oos["NIV"], oos["stress_ahead"], "NIV")
recall_nfci, precision_nfci = get_metrics(oos["NFCI"], oos["stress_ahead"], "NFCI")

# Print maximal results
print("\n=== MAXIMAL RIGOROUS OOS STRESS COMPARISON ===")
print(f"Periods: {len(oos)}")
print("\nAUC-ROC:")
print(f"  NIV v6: {auc_niv:.4f}")
print(f"  NFCI: {auc_nfci:.4f}")
print(f"  STLFSI: {auc_stlfsi:.4f}")
print(f"  KCFSI: {auc_kcfsi:.4f}")
print(f"  VIX: {auc_vix:.4f}")

print("\nPR-AUC:")
print(f"  NIV v6: {pr_niv:.4f}")
print(f"  NFCI: {pr_nfci:.4f}")

print("\nBootstrap p-values (NIV vs benchmark, p = Prob(NIV > benchmark)):")
print(f"  vs NFCI: {p_vs_nfci:.4f}")
print(f"  vs KCFSI: {p_vs_kcfsi:.4f}")
print(f"  vs VIX: {p_vs_vix:.4f}")

print("\nOptimal F1 Metrics:")
print(f"  NIV Recall: {recall_niv:.1%}, Precision: {precision_niv:.1%}")
print(f"  NFCI Recall: {recall_nfci:.1%}, Precision: {precision_nfci:.1%}")

# Save plots
plt.figure(figsize=(12,8))
for name, probs, color in [("NIV", -oos["NIV"], "blue"), ("NFCI", oos["NFCI"], "red")]:
    precision, recall, _ = precision_recall_curve(oos["stress_ahead"], probs)
    plt.plot(recall, precision, label=f"{name} (PR-AUC = {auc(recall, precision):.3f})", color=color)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves - NIV vs NFCI")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("pr_curve_niv_vs_nfci.png")
plt.close()

oos.to_csv("max_rigorous_comparison.csv")
print("\nSaved PR curve plot and full data.")