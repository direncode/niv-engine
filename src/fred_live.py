import pandas as pd, os
from fredapi import Fred

fred = Fred(api_key=os.getenv("FRED_API_KEY"))
series = {
    "GDP": "GDP", "M2": "M2SL", "GOVT_INVEST": "GPDIC1",
    "FEDFUNDS": "FEDFUNDS", "DGS10": "DGS10",
    "TB3MS": "TB3MS", "UNRATE": "UNRATE"
}

df = pd.DataFrame({k: fred.get_series(v) for k, v in series.items()})
df.index = pd.to_datetime(df.index)
df = df.resample("M").ffill()
os.makedirs("data", exist_ok=True)
df.to_csv("data/fred_live.csv")
print("✅ Saved live FRED data to data/fred_live.csv")


