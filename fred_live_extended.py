# fred_live_extended.py
# Final maximal FRED fetch for NIV supremacy
# Core + benchmarks + credit + lending + high-yield OAS

import pandas as pd
from fredapi import Fred
import os

fred = Fred(api_key=os.getenv("120ef46bc7d037a848ae2fb057644064"))

if fred is None:
    raise ValueError("FRED_API_KEY not set. Use $env:FRED_API_KEY = 'your_key' in PowerShell.")

# Core NIV series
core_series = {
    "GDPC1": "GDPC1",           # Real GDP
    "GPDIC1": "GPDIC1",         # Real private investment
    "M2SL": "M2SL",             # M2
    "FEDFUNDS": "FEDFUNDS",     # Fed funds rate
    "DGS10": "DGS10",           # 10-year Treasury
    "TB3MS": "TB3MS",           # 3-month Treasury
    "TCU": "TCU",               # Capacity utilization
    "GFDEBTN": "GFDEBTN"        # Federal debt
}

# Fed stress benchmarks
benchmarks = {
    "NFCI": "NFCI",             # Chicago Fed National Financial Conditions Index
    "STLFSI": "STLFSI4",        # St. Louis Fed Financial Stress Index
    "KCFSI": "KCFSI",           # Kansas City Fed Financial Stress Index
    "VIX": "VIXCLS"             # CBOE Volatility Index
}

# Supremacy friction enhancers
supremacy = {
    "BAA10Y": "BAA10Y",                     # Moody's BAA corporate yield
    "AAA10Y": "AAA10Y",                     # Moody's AAA corporate yield
    "DRTSCILM": "DRTSCILM",                 # Senior Loan Officer Survey - tighter lending
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",         # ICE BofA US High Yield OAS
    "BAMLC0A4CBBB": "BAMLC0A4CBBB"           # ICE BofA BBB Corporate OAS
}

series = {**core_series, **benchmarks, **supremacy}

print("Fetching maximal extended FRED series for NIV supremacy...")
data = {}
for name, sid in series.items():
    try:
        s = fred.get_series(sid)
        data[name] = s
        print(f"  ✓ {name} ({sid}): {len(s)} points, {s.index[0].date()} → {s.index[-1].date()}")
    except Exception as e:
        print(f"  ✗ Failed {name} ({sid}): {e}")

df = pd.DataFrame(data)
df.index = pd.to_datetime(df.index)

# Resample to quarterly means (rates/indexes) — levels like GDP are quarterly points
df_quarterly = df.resample("QS").mean()

df_quarterly.index.name = "DATE"

output_file = "fred_live_extended.csv"
df_quarterly.to_csv(output_file)

print(f"\n✅ Maximal extended quarterly data saved to: {output_file}")
print(f"   Shape: {df_quarterly.shape}")
print(f"   Date range: {df_quarterly.index.min().date()} → {df_quarterly.index.max().date()}")
print("   Columns:", ", ".join(df_quarterly.columns.tolist()))