import hopsworks
import joblib, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ── Login to Hopsworks ───────────────────────────────
project = hopsworks.login(
    project="TENPSAQI_Ptoject",   # your project name
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key = os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

# ── Query Feature Group ──────────────────────────────
aqi_fg = fs.get_feature_group("pakistan_aqi_features", version=1)
df = aqi_fg.read()

# ── Select Features & Target ─────────────────────────
X = df.drop(columns=["aqi", "aqi_label", "timestamp", "city"])  # inputs
y = df["aqi"]                                                   # target

# ── Train/Test Split ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train Random Forest ──────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Yayy! Random Forest trained")
print("MSE:", mse)
print("R²:", r2)

# ── Save & Register Model ────────────────────────────
mr = project.get_model_registry()
model_dir = "aqi_rf_model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_dir + "/model.pkl")

aqi_model = mr.python.create_model(
    name="aqi_random_forest",
    metrics={"mse": mse, "r2": r2},
    description="Random Forest AQI predictor"
)

aqi_model.save(model_dir)   # <-- correct way
print("Yayy!Model registered in Hopsworks")
