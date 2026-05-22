import os
import time
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from pymongo import MongoClient

# ── Config ──────────────────────────────────────────────
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]   # set in GitHub Secrets
MONGO_URI           = os.environ["MONGO_URI"]             # set in GitHub Secrets
PAKISTAN_TZ         = ZoneInfo("Asia/Karachi")
OUTPUT_FILE         = "pakistan_aqi_data.csv"

client = MongoClient(MONGO_URI)
db = client["aqi_db"]
collection = db["pakistan_aqi"]

CITIES = [
    {"name": "Karachi",    "lat": 24.8607, "lon": 67.0011},
    {"name": "Lahore",     "lat": 31.5497, "lon": 74.3436},
    {"name": "Islamabad",  "lat": 33.6844, "lon": 73.0479},
    {"name": "Faisalabad", "lat": 31.4504, "lon": 73.1350},
    {"name": "Peshawar",   "lat": 34.0151, "lon": 71.5249},
    {"name": "Multan",     "lat": 30.1575, "lon": 71.5249},
    {"name": "Quetta",     "lat": 30.1798, "lon": 66.9750},
]

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}


# ── Fetch AQI ───────────────────────────────────────────
def fetch_aqi(city: dict):
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={city['lat']}&lon={city['lon']}&appid={OPENWEATHER_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[ERROR] {city['name']}: {e}")
        return None

    if not data.get("list"):
        print(f"[WARN] {city['name']}: empty response")
        return None

    item = data["list"][0]
    aqi  = item["main"]["aqi"]
    comp = item["components"]

    utc_now   = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    local_now = utc_now.astimezone(PAKISTAN_TZ)

    return {
        "timestamp_pk": local_now,
        "city":         city["name"],
        "lat":          city["lat"],
        "lon":          city["lon"],
        "aqi":          aqi,
        "aqi_label":    AQI_LABELS.get(aqi, "Unknown"),
        "co":           comp.get("co"),
        "no":           comp.get("no"),
        "no2":          comp.get("no2"),
        "o3":           comp.get("o3"),
        "so2":          comp.get("so2"),
        "pm2_5":        comp.get("pm2_5"),
        "pm10":         comp.get("pm10"),
        "nh3":          comp.get("nh3"),
    }


# ── Feature engineering ──────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["timestamp_pk"])
    df["hour"]        = dt.dt.hour
    df["day"]         = dt.dt.day
    df["month"]       = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)

    df = df.sort_values(["city", "timestamp_pk"])
    df["aqi_lag1"]   = df.groupby("city")["aqi"].shift(1)
    df["aqi_change"] = df["aqi"] - df["aqi_lag1"]

    return df


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Collect data
    rows = []
    for city in CITIES:
        row = fetch_aqi(city)
        if row:
            rows.append(row)
        time.sleep(0.5)

    if not rows:
        print("No data collected this run.")
        exit(1)

    # 2. Build DataFrame + engineer features
    df = pd.DataFrame(rows)
    df = add_features(df)

    # 3. Clean primary keys
    df = df.dropna(subset=["city", "timestamp_pk"])
    df["city"]         = df["city"].astype(str).str.strip()
    df["timestamp_pk"] = pd.to_datetime(df["timestamp_pk"])

    # 4. Save locally (optional history)
    file_exists = os.path.isfile(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode="a", header=not file_exists, index=False)
    print(f" Saved {len(df)} rows → {OUTPUT_FILE}")

    # 5. Insert into MongoDB
    try:
        records = df.to_dict("records")
        collection.insert_many(records)
        print(f" Uploaded {len(records)} rows → MongoDB Atlas")
    except Exception as e:
        print(f" Insert failed: {e}")
        exit(1)
