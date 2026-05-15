import pandas as pd

# ── Load raw dataset ──────────────────────────────────────────────
df = pd.read_csv("C:/Users/Admin/Downloads/pakistan_aqi_data.csv")

# ── Handle timestamps ─────────────────────────────────────────────
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

# ── Deduplicate ───────────────────────────────────────────────────
df = df.drop_duplicates(subset=['city','timestamp'], keep='first')

# ── Sort by city + time ───────────────────────────────────────────
df = df.sort_values(['city','timestamp'])

# ── Add lag features ──────────────────────────────────────────────
df['aqi_lag1'] = df.groupby('city')['aqi'].shift(1)
df['aqi_change'] = df['aqi'] - df['aqi_lag1']

# Drop rows where lag features are NaN (first entry per city)
df = df.dropna(subset=['aqi_lag1','aqi_change'])

# ── Save engineered dataset ───────────────────────────────────────
df.to_csv("C:/Users/Admin/Downloads/pakistan_aqi_data_engineered.csv", index=False)
print("Cleaned + engineered dataset saved → pakistan_aqi_data_engineered.csv")
print(f"Total rows after cleaning: {len(df)}")
