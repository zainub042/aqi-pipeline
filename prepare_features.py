import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import time
import hopsworks

# ── Config ──────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")   # GitHub Secret
PAKISTAN_TZ = ZoneInfo("Asia/Karachi")

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

# ── Fetch AQI for one city ──────────────────────────────
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

    utc_now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    local_now = utc_now.astimezone(PAKISTAN_TZ)

    return {
        "timestamp_pk":  local_now,   # store as datetime, not string
        "city":       city["name"],
        "lat":        city["lat"],
        "lon":        city["lon"],
        "aqi":        aqi,
        "aqi_label":  AQI_LABELS.get(aqi, "Unknown"),
        "co":         comp.get("co"),
        "no":         comp.get("no"),
        "no2":        comp.get("no2"),
        "o3":         comp.get("o3"),
        "so2":        comp.get("so2"),
        "pm2_5":      comp.get("pm2_5"),
        "pm10":       comp.get("pm10"),
        "nh3":        comp.get("nh3"),
    }

# ── Add time + lag features ──────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["timestamp_pk"])
    df["hour"]        = dt.dt.hour
    df["day"]         = dt.dt.day
    df["month"]       = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)

    # Lag features per city
    df = df.sort_values(["city","timestamp_pk"])
    df["aqi_lag1"] = df.groupby("city")["aqi"].shift(1)
    df["aqi_change"] = df["aqi"] - df["aqi_lag1"]
    df = df.dropna(subset=["aqi_lag1","aqi_change"])
    return df

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    rows = []
    for city in CITIES:
        row = fetch_aqi(city)
        if row:
            rows.append(row)
        time.sleep(0.5)

    if not rows:
        print("No data collected this run.")
        exit()

    df = pd.DataFrame(rows)
    df = add_features(df)

    # Ensure timestamp is proper datetime
    df["timestamp_pk"] = pd.to_datetime(df["timestamp_pk"])

    # ── Upload to Hopsworks Feature Store ─────────────────
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("BOOO! No HOPSWORKS_API_KEY found in environment variables")

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    aqi_fg = fs.get_or_create_feature_group(
        name="pakistan_aqi_features",
        version=1,
        primary_key=["city","timestamp_pk"],
        description="Engineered AQI dataset with lag + time features",
        online_enabled=True
    )

    try:
        aqi_fg.insert(df, write_options={"wait_for_job": True})
        print(f"Yayyy!Uploaded {len(df)} rows with {df.shape[1]} columns to Hopsworks Feature Store")
    except Exception as e:
        print("SORRY! Insert failed:", e)
