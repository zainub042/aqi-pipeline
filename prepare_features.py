import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import time
import os
import hopsworks
from hsfs import feature

# ── Config ──────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY   = os.getenv("HOPSWORKS_API_KEY")
PAKISTAN_TZ = ZoneInfo("Asia/Karachi")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "pakistan_aqi_data.csv")

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

# ── Fetch AQI ──────────────────────────────────────────────
def fetch_aqi(city: dict):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={city['lat']}&lon={city['lon']}&appid={OPENWEATHER_API_KEY}"
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
        "timestamp_pk": local_now,
        "city": city["name"],
        "lat": city["lat"],
        "lon": city["lon"],
        "aqi": aqi,
        "aqi_label": AQI_LABELS.get(aqi, "Unknown"),
        "co": comp.get("co"),
        "no": comp.get("no"),
        "no2": comp.get("no2"),
        "o3": comp.get("o3"),
        "so2": comp.get("so2"),
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "nh3": comp.get("nh3"),
    }

# ── Feature engineering ──────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["timestamp_pk"])
    df["hour"]        = dt.dt.hour
    df["day"]         = dt.dt.day
    df["month"]       = dt.dt.month
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)

    df = df.sort_values(["city","timestamp_pk"])
    df["aqi_lag1"] = df.groupby("city")["aqi"].shift(1)
    df["aqi_change"] = df["aqi"] - df["aqi_lag1"]

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

    # Clean PKs
    df = df.dropna(subset=["city","timestamp_pk"])
    df["city"] = df["city"].astype(str).str.strip()
    df["timestamp_pk"] = pd.to_datetime(df["timestamp_pk"])

    # Save locally (for debugging only)
    file_exists = os.path.isfile(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode="a", header=not file_exists, index=False)
    print(f"✅ Saved {len(df)} rows → {OUTPUT_FILE}")

    # ── Force correct dtypes for Hopsworks ──────────────────────────────
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")
    df["aqi"] = df["aqi"].astype("int32")
    df["co"] = df["co"].astype("float32")
    df["no"] = df["no"].astype("float32")
    df["no2"] = df["no2"].astype("float32")
    df["o3"] = df["o3"].astype("float32")
    df["so2"] = df["so2"].astype("float32")
    df["pm2_5"] = df["pm2_5"].astype("float32")
    df["pm10"] = df["pm10"].astype("float32")
    df["nh3"] = df["nh3"].astype("float32")
    df["hour"] = df["hour"].astype("int32")
    df["day"] = df["day"].astype("int32")
    df["month"] = df["month"].astype("int32")
    df["day_of_week"] = df["day_of_week"].astype("int32")
    df["is_weekend"] = df["is_weekend"].astype("int32")
    df["aqi_lag1"] = df["aqi_lag1"].astype("float32")
    df["aqi_change"] = df["aqi_change"].astype("float32")

    # ── Upload to Hopsworks ──────────────────────────────
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    features = [
        feature.Feature("city", "STRING"),
        feature.Feature("timestamp_pk", "TIMESTAMP"),
        feature.Feature("lat", "FLOAT"),
        feature.Feature("lon", "FLOAT"),
        feature.Feature("aqi", "INT"),
        feature.Feature("aqi_label", "STRING"),
        feature.Feature("co", "FLOAT"),
        feature.Feature("no", "FLOAT"),
        feature.Feature("no2", "FLOAT"),
        feature.Feature("o3", "FLOAT"),
        feature.Feature("so2", "FLOAT"),
        feature.Feature("pm2_5", "FLOAT"),
        feature.Feature("pm10", "FLOAT"),
        feature.Feature("nh3", "FLOAT"),
        feature.Feature("hour", "INT"),
        feature.Feature("day", "INT"),
        feature.Feature("month", "INT"),
        feature.Feature("day_of_week", "INT"),
        feature.Feature("is_weekend", "INT"),
        feature.Feature("aqi_lag1", "FLOAT"),
        feature.Feature("aqi_change", "FLOAT"),
    ]

    # Create once, otherwise get existing
    try:
        aqi_fg = fs.get_feature_group(name="pakistan_aqi_features", version=2)
    except:
        aqi_fg = fs.create_feature_group(
            name="pakistan_aqi_features",
            version=2,
            primary_key=["city","timestamp_pk"],
            description="Pakistan AQI dataset v2 with lag + time features (online enabled)",
            online_enabled=True,
            features=features
        )

    try:
        aqi_fg.insert(df, write_options={"wait_for_job": True, "online": True})
        print(f"🚀 Uploaded {len(df)} rows with {df.shape[1]} columns to Hopsworks Feature Store v2 (offline + online)")
    except Exception as e:
        print("❌ Insert failed:", e)
