"""
Training Pipeline — Pakistan AQI Prediction
Models: Random Forest, Ridge Regression, Gradient Boosting
Loads dataset from MongoDB Atlas
Saves trained models into MongoDB GridFS
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pymongo import MongoClient
import gridfs
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI")
MODEL_DIR = "aqi_models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ['pm2_5','pm10','so2','o3','co','nh3','hour','day_of_week','is_weekend','aqi_lag1']
TARGET   = 'aqi'

# ── MongoDB setup ────────────────────────────────────────
client     = MongoClient(MONGO_URI)
db         = client["aqi_db"]
collection = db["pakistan_aqi"]
fs_grid    = gridfs.GridFS(db)


# ── Helper: evaluate ─────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    mae  = round(float(mean_absolute_error(y_true, y_pred)), 4)
    r2   = round(float(r2_score(y_true, y_pred)), 4)
    print(f"\n{'='*40}\n{name}\n{'='*40}")
    print(f"RMSE: {rmse} | MAE: {mae} | R²: {r2}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_eval_plot(name, y_true, y_pred, metrics, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=15, color='steelblue')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    axes[0].set_title(f'{name} — Actual vs Predicted')
    axes[0].set_xlabel('Actual AQI')
    axes[0].set_ylabel('Predicted AQI')

    # Residuals
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=40, color='#e74c3c', alpha=0.7)
    axes[1].axvline(0, color='black', linewidth=1.5, linestyle='--')
    axes[1].set_title(f'{name} — Residuals')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot → {path}")
    return path


def save_model(model, filename, metrics, best_model=False):
    """Save model to disk and GridFS, deleting old versions first."""
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)

    #  Delete old versions to avoid duplicates
    for old_file in fs_grid.find({"filename": filename}):
        fs_grid.delete(old_file._id)

    with open(path, "rb") as f:
        file_id = fs_grid.put(
            f,
            filename=filename,
            metrics=metrics,
            trained_at=datetime.utcnow().isoformat(),
            best_model=best_model
        )
    print(f" {filename} stored in MongoDB GridFS with id {file_id}")
    return path


# ── Step 1: Load dataset from MongoDB ────────────────────
print("Loading dataset from MongoDB Atlas...")
cutoff = datetime.utcnow() - timedelta(days=90)
cursor = collection.find({"timestamp_pk": {"$gte": cutoff}}, {"_id": 0})
df = pd.DataFrame(list(cursor))

if df.empty:
    raise ValueError("No data found in MongoDB for the last 30 days!")

df['timestamp_pk'] = pd.to_datetime(df['timestamp_pk'], utc=True)
df = df.sort_values(['city', 'timestamp_pk']).reset_index(drop=True)
print(f"Loaded {len(df)} rows across {df['city'].nunique()} cities")

# ── Step 2: Feature engineering ──────────────────────────
print("\nEngineering features...")

# ✅ Fix aqi_lag1 NaN — use current aqi as fallback
df['aqi_lag1'] = df.groupby('city')['aqi'].shift(1)
df['aqi_lag1'] = df['aqi_lag1'].fillna(df['aqi'])

# Add time features if missing
if 'hour' not in df.columns:
    df['hour'] = pd.to_datetime(df['timestamp_pk']).dt.hour
if 'day_of_week' not in df.columns:
    df['day_of_week'] = pd.to_datetime(df['timestamp_pk']).dt.dayofweek
if 'is_weekend' not in df.columns:
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

df = df.dropna(subset=FEATURES + [TARGET])
print(f"After dropna: {len(df)} rows")

# ── Step 3: Preprocess ───────────────────────────────────
print("\nPreprocessing...")
for col in ['pm2_5', 'pm10', 'co']:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)
    print(f"Capped {col} at {cap:.2f}")

X = df[FEATURES].values
y = df[TARGET].values.astype(float)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"Train={len(X_train)}, Test={len(X_test)}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

for old in fs_grid.find({"filename": "scaler.pkl"}):
    fs_grid.delete(old._id)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    fs_grid.put(f, filename="scaler.pkl", trained_at=datetime.utcnow().isoformat())
print(" Scaler saved to MongoDB")

# ── Step 4: Train Models ──────────────────────────────────
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=200, max_depth=15,
    min_samples_leaf=2, n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
metrics_rf = evaluate("Random Forest", y_test, rf.predict(X_test))
save_eval_plot("Random Forest", y_test, rf.predict(X_test), metrics_rf, "rf_eval.png")

print("\nTraining Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
metrics_ridge = evaluate("Ridge Regression", y_test, ridge.predict(X_test_sc))
save_eval_plot("Ridge Regression", y_test, ridge.predict(X_test_sc), metrics_ridge, "ridge_eval.png")

print("\nTraining Gradient Boosting...")
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=5,
    learning_rate=0.05, random_state=42
)
gb.fit(X_train, y_train)
metrics_gb = evaluate("Gradient Boosting", y_test, gb.predict(X_test))
save_eval_plot("Gradient Boosting", y_test, gb.predict(X_test), metrics_gb, "gb_eval.png")

# ── Step 5: Compare & pick best ──────────────────────────
all_metrics = {
    "Random Forest":    metrics_rf,
    "Ridge Regression": metrics_ridge,
    "Gradient Boosting": metrics_gb
}
best_name = min(all_metrics, key=lambda k: all_metrics[k]['rmse'])
print(f"\n🏆 Best model: {best_name}")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
models = list(all_metrics.keys())
for i, metric in enumerate(['rmse', 'mae', 'r2']):
    vals   = [all_metrics[m][metric] for m in models]
    colors = ['#2ecc71' if m == best_name else '#4c72b0' for m in models]
    axes[i].bar(models, vals, color=colors, edgecolor='white')
    axes[i].set_title(metric.upper())
    axes[i].tick_params(axis='x', rotation=15)
plt.suptitle('Model Comparison (🟢 = best)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150)
plt.close()
print(" Comparison chart saved")

# ── Step 6: Save Models to MongoDB GridFS ────────────────
filename_map = {
    "Random Forest":     "random_forest.pkl",
    "Ridge Regression":  "ridge_regression.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
}
model_map = {
    "Random Forest":     rf,
    "Ridge Regression":  ridge,
    "Gradient Boosting": gb,
}
metrics_map = {
    "Random Forest":     metrics_rf,
    "Ridge Regression":  metrics_ridge,
    "Gradient Boosting": metrics_gb,
}

for name, model in model_map.items():
    save_model(
        model,
        filename_map[name],
        metrics_map[name],
        best_model=(name == best_name)
    )

print("\nYAYY! All models trained and stored in MongoDB GridFS")
print(f"🏆 Best model: {best_name} (RMSE: {all_metrics[best_name]['rmse']})")