# pip install fredapi pandas numpy
import os, pandas as pd, numpy as np
from fredapi import Fred

API_KEY = os.getenv("FRED_API_KEY") or "YOUR_FRED_KEY"
fred = Fred(api_key=API_KEY)

# Valid, active FRED series (monthly or quarterly)
SERIES = {
    "GDP":        ("GDPMC1", "q"),          # Real GDP
    "GOVT_INVEST":("GPDIC1","q"),           # Gross private domestic investment (broad fiscal proxy)
    "M2":         ("M2SL","m"),             # Money supply
    "CAPU":       ("TCU","m"),              # Capacity utilization
    "FEDFUNDS":   ("FEDFUNDS","m"),         # Fed Funds
    "DGS10":      ("DGS10","m"),            # 10-year Treasury yield
    "TB3MS":      ("TB3MS","m"),            # 3-month T-bill
    "UNRATE":     ("UNRATE","m"),           # Unemployment
    "EDU_INV":    ("DGDSRXH1M027SBEA","q")  # Educational investment proxy
}


def fetch_series(sid, freq):
    s = fred.get_series(sid)
    s.name = sid
    s = s.to_frame()
    s.index = pd.to_datetime(s.index)
    if freq == "m":
        return s.asfreq("MS")
    m = s.asfreq("QS").asfreq("MS").ffill()
    return m.interpolate()

frames = []
for _, (sid, freq) in SERIES.items():
    try:
        frames.append(fetch_series(sid, freq))
    except Exception as e:
        print(f"⚠️ Skipped {sid}: {e}")

df = pd.concat(frames, axis=1).sort_index()
df = df.loc["1947-01-01":].apply(pd.to_numeric, errors="coerce").dropna(how="all")
df.to_csv("fred_live.csv", index_label="DATE")
print("✅ Saved fred_live.csv with shape:", df.shape)

