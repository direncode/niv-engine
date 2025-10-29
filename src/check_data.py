import pandas as pd
df = pd.read_csv("niv_timeseries.csv", parse_dates=["DATE"]).set_index("DATE")

print("Coverage:", df.index.min().date(), "→", df.index.max().date())
print("\nNon-null counts:\n", df.notna().sum())
print("\nNIV summary:\n", df["NIV_t"].describe())

