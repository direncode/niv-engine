# niv_final_legendary_test.py
# The ONE test that ends all tests — Treasury-level rigor

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import yaml

# Config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)["params"]

ALPHA1, ALPHA2, ALPHA3 = cfg["alpha1"], cfg["alpha2"], cfg["alpha3"]
LAMBDA, ETA = cfg["lambda"], cfg["eta"]
WINDOW = 12

# Load data
df = pd.read_csv("fred_live_extended.csv", parse_dates=["DATE"], index_col="DATE")

# Your 20 legendary events — exact peaks
events = pd.to_datetime([
    "1966-07-01","1970-01-01","1974-01-01","1980-04-01","1982-01-01",
    "1987-10-01","1991-01-01","1998-10-01","2001-04-01","2008-10-01",
    "2011-10-01","2016-01-01","2018-04-01","2018-10-01","2019-10-01",
    "2020-04-01","2022-04-01","2023-04-01","2025-01-01","2025-04-01"
])

# Stress = 1 if within ±4 quarters of any event (your chart window)
stress = pd.Series(0, index=df.index)
for e in events:
    if e in df.index:
        window = df.loc[e-pd.DateOffset(months=12):e+pd.DateOffset(months=12)]
        stress.loc[window.index] = 1

# 2-year ahead target
y = stress.shift(-8).rolling(8).max().fillna(0)

# Kernel
kernel = np.exp(-LAMBDA * np.arange(WINDOW)); kernel /= kernel.sum()
def discounted(s): 
    a = np.concatenate([np.zeros(WINDOW-1), s.fillna(0)])
    return pd.Series(np.convolve(a, kernel, 'valid'), index=s.index)

# Expanding window OOS
results = []
for end in range(120, len(df)-8):
    train = df.iloc[:end]
    date = df.index[end]
    d = train.copy().interpolate().ffill().bfill()
    
    # Full NIV v6
    d["ΔG"] = d["GPDIC1"].pct_change().clip(-0.5,0.5)
    d["ΔA"] = d["M2SL"].pct_change(12).rolling(3).mean().clip(-0.5,0.5)
    d["Δr"] = d["FEDFUNDS"].diff()/100
    d["u"] = discounted(ALPHA1*d["ΔG"] + ALPHA2*d["ΔA"] + ALPHA3*d["Δr"])
    d["P"] = d["GPDIC1"]/d["GDPC1"]
    d["X"] = (1-d["TCU"]/100).clip(0,1)
    d["F"] = (d["DGS10"]-d["TB3MS"]).clip(lower=0) + d["GFDEBTN"]/(d["GDPC1"]*4)
    d["NIV"] = (d["u"] * d["P"]**2) / (d["X"]+d["F"]).replace(0,1e-9)**ETA
    d["NIV"] = d["NIV"].replace([np.inf,-np.inf],np.nan).fillna(0)
    
    results.append({
        "date": date,
        "NIV": d["NIV"].iloc[-1],
        "NFCI": d["NFCI"].iloc[-1],
        "stress_2y": y.loc[date]
    })

oos = pd.DataFrame(results).set_index("date").dropna()

# PR-AUC
pr_auc_niv = auc(*precision_recall_curve(oos["stress_2y"], -oos["NIV"])[1::-1])
pr_auc_nfci = auc(*precision_recall_curve(oos["stress_2y"], oos["NFCI"])[1::-1])

# Precision at 90% recall (the Treasury question: "When you catch 90% of crises, how many false alarms?")
def prec_at_recall(probs, y_true, target_recall=0.90):
    p, r, t = precision_recall_curve(y_true, probs)
    idx = np.where(r >= target_recall)[0]
    return p[idx[-1]] if len(idx)>0 else 0

p90_niv = prec_at_recall(-oos["NIV"], oos["stress_2y"])
p90_nfci = prec_at_recall(oos["NFCI"], oos["stress_2y"])

# Bootstrap NIV vs NFCI
boot = []
for _ in range(2000):
    s = oos.sample(frac=1, replace=True)
    if s["stress_2y"].nunique()>1:
        a_niv = auc(*precision_recall_curve(s["stress_2y"], -s["NIV"])[1::-1])
        a_nfci = auc(*precision_recall_curve(s["stress_2y"], s["NFCI"])[1::-1])
        boot.append(a_niv - a_nfci)
p_vs_nfci = np.mean(np.array(boot) >= 0)

# Final output
print("\nLEGENDARY FINAL TEST — 2-YEAR AHEAD STRESS PREDICTION")
print(f"Periods: {len(oos)} | Expanding window | Your 20 events")
print(f"NIV PR-AUC: {pr_auc_niv:.4f}")
print(f"NFCI PR-AUC: {pr_auc_nfci:.4f}")
print(f"Precision at 90% recall — NIV: {p90_niv:.1%} | NFCI: {p90_nfci:.1%}")
print(f"Bootstrap p-value (NIV ≥ NFCI): {p_vs_nfci:.4f}")

plt.figure(figsize=(10,7))
for name, probs, color in [("NIV", -oos["NIV"], "blue"), ("NFCI", oos["NFCI"], "red")]:
    p,r,_ = precision_recall_curve(oos["stress_2y"], probs)
    plt.plot(r, p, label=f"{name} (PR-AUC={auc(r,p):.3f})", lw=2, color=color)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Definitive PR Curve — NIV vs NFCI")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("definitive_pr_curve.png", dpi=300, bbox_inches="tight")

print("Saved definitive PR curve.")