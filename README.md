# 🌫️ AQI Forecasting System

> **An end-to-end MLOps pipeline for real-time Air Quality Index prediction across 7 cities — powered by OpenWeather API, MongoDB, GitHub Actions, and Streamlit.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aqi-pipeline-crgkkiixn4gzgse8s6j6oy.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=github-actions)](https://github.com/features/actions)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?logo=mongodb)](https://www.mongodb.com/)

---

## 🚀 Live Demo

**👉 [https://aqi-pipeline-crgkkiixn4gzgse8s6j6oy.streamlit.app/](https://aqi-pipeline-crgkkiixn4gzgse8s6j6oy.streamlit.app/)**

The dashboard provides:
- 📊 Real-time **3-day AQI forecasts** for 7 cities
- 🔍 **SHAP feature importance** explanations
- ⚠️ **Hazardous AQI alerts** with health warnings
- 📈 Historical AQI trend visualization

---

## 🌐 Overview

This project builds a complete **MLOps pipeline** for AQI (Air Quality Index) forecasting. It automatically collects pollutant data from the **OpenWeather Air Pollution API** every hour across 7 cities, stores processed features in **MongoDB**, retrains machine learning models daily via **GitHub Actions**, and serves live predictions through an interactive **Streamlit** dashboard with SHAP explainability and hazardous level alerts.

The project was originally scoped for a single country but was extended to cover **7 cities** for broader real-world applicability.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔄 **Automated Feature Pipeline** | Hourly data ingestion from OpenWeather API via GitHub Actions |
| 🗄️ **MongoDB Feature Store** | Cloud-hosted, duplicate-free, Asia/Karachi timestamped storage |
| 📅 **90-Day Historical Backfill** | Comprehensive training dataset across all 7 cities |
| 📊 **Exploratory Data Analysis** | Trend analysis, correlation studies, outlier detection |
| 🤖 **Multiple ML Models** | Random Forest, Ridge Regression, and LSTM (TensorFlow) |
| 🎯 **3-Day Forecasting** | Next 72-hour AQI predictions per city |
| 🔍 **SHAP Explainability** | Feature importance analysis for every prediction |
| ⚠️ **Hazard Alerts** | Automatic warnings for dangerous AQI levels |
| ⚙️ **Full CI/CD Automation** | GitHub Actions for hourly + daily pipeline runs |
| 🌍 **7-City Coverage** | Multi-city monitoring and forecasting dashboard |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **ML / DL** | Scikit-learn, TensorFlow / Keras (LSTM) |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib |
| **Data Storage** | MongoDB (Cloud) |
| **Data Source** | OpenWeather Air Pollution API |
| **Automation / CI/CD** | GitHub Actions |
| **Dashboard** | Streamlit |
| **Dev Environment** | Anaconda Navigator, Dev Container |
| **Version Control** | Git / GitHub |
| **Config** | python-dotenv |

> **Note:** Flask is **not** used in this project. Streamlit is the sole front-end and serving layer.

---


---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   GITHUB ACTIONS (CI/CD)                 │
│                                                          │
│  ┌───────────────────────┐   ┌───────────────────────┐   │
│  │   Feature Pipeline    │   │   Training Pipeline   │   │
│  │   (Every 2hr)         │   │   (Every Day)         │   │
│  │   fpipeline.yml       │   │   train_model.yml     │   │
│  └──────────┬────────────┘   └───────────┬───────────┘   │
└─────────────│───────────────────────────│───────────────-┘
              │                           │
              ▼                           │
   ┌──────────────────────┐               │
   │   OpenWeather API    │               │
   │   (7 Cities)         │               │
   └──────────┬───────────┘               │
              │  Raw pollutant data       │
              ▼                           │
   ┌──────────────────────┐               │
   │  prepare_features.py │               │
   │  - Time features     │               │
   │  - AQI change rate   │               │
   │  - Deduplication     │               │
   │  - Asia/Karachi TZ   │               │
   └──────────┬───────────┘               │
              │                           │
              ▼                           ▼
   ┌────────────────────────────────────────────┐
   │              MongoDB Feature Store         │
   │     (90+ days · 7 cities · PST/Karachi)    │
   └───────────────────────┬────────────────────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │       q.py          │
                 │  Random Forest      │
                 │  Ridge Regression   │
                 │  LSTM (local only)  │
                 └──────────┬──────────┘
                            │
                            ▼
               ┌────────────────────────┐
               │        app.py          │
               │   Streamlit Dashboard  │
               │  - 3-day forecasts     │
               │  - SHAP analysis       │
               │  - Hazard alerts       │
               │  - 7-city selector     │
               └────────────────────────┘
```

---

## 🌍 Cities Covered

The system monitors and forecasts AQI for **7 cities** simultaneously. Each city is fetched independently every hour, tagged with a city identifier, and stored in MongoDB. The Streamlit dashboard lets users switch between cities to view individual forecasts.

> *(Add your 7 city names here)*

---

## 📊 AQI Scale

This project uses the **OpenWeather Air Pollution API**, which returns AQI on a **1–5 scale** (not the standard 0–500 scale). The dashboard includes a built-in reference guide.

| AQI Value | Classification | Health Implication |
|:---------:|---------------|-------------------|
| 1 | 🟢 Good | Air quality is satisfactory; little or no risk |
| 2 | 🟡 Fair | Acceptable; minor concern for sensitive individuals |
| 3 | 🟠 Moderate | Unhealthy for sensitive groups |
| 4 | 🔴 Poor | Everyone may begin to experience health effects |
| 5 | 🟣 Very Poor | Health alert — serious risk for the entire population |

> Conversion to the standard 0–500 AQI scale is planned as future work.

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.9+
- MongoDB Atlas account (free tier works)
- OpenWeather API key ([free tier](https://openweathermap.org/api/air-pollution))
- GitHub account (for Actions automation)

### 1. Clone the Repository

```bash
git clone https://github.com/Zainub042/aqi-forecasting.git
cd aqi-forecasting
```

### 2. Create a Virtual Environment

```bash
conda create -n aqi_env python=3.9
conda activate aqi_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
OPENWEATHER_API_KEY=your_openweather_api_key
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB_NAME=your_database_name
MONGODB_COLLECTION=your_collection_name
```

> For GitHub Actions, add these same values as **repository secrets** under `Settings → Secrets and variables → Actions`.

### 5. Prepare Features and Backfill Historical Data

```bash
python prepare_features.py
```

### 6. Train the Models

```bash
python q.py
```

### 7. Launch the Dashboard Locally

```bash
streamlit run app.py
```

---

## 📄 File Descriptions

### `app.py` — Streamlit Dashboard
The main web application. Loads the trained model and latest features from MongoDB, computes 3-day AQI forecasts for each city, and renders:
- Interactive forecast charts per city
- SHAP feature importance visualizations
- Hazardous AQI level alerts with health guidance
- AQI scale reference guide

### `prepare_features.py` — Feature Engineering
Handles all data preparation:
- Fetches raw pollutant data from the OpenWeather API for all 7 cities
- Engineers time-based features (hour, day of week, month)
- Computes the AQI change rate (current − previous reading)
- Deduplicates records before writing to MongoDB
- Converts all timestamps to **Asia/Karachi** timezone

### `q.py` — Training Script
Trains and evaluates machine learning models:
- Retrieves feature data from MongoDB
- Trains **Random Forest**, **Ridge Regression**, and **Gradient Boosting** models
- Evaluates each model using RMSE, MAE, and R²
- Saves the best-performing model for use in `app.py`

### `requirements.txt` — Dependencies

Key packages include:

```
streamlit==1.32.0
plotly==5.19.0
pandas==2.2.2
numpy==1.26.4
pymongo==4.6.3
dnspython==2.6.1
scikit-learn==1.6.1
joblib==1.4.2
shap==0.45.0
streamlit-autorefresh==1.0.1
requests==2.31.0
matplotlib
python-dotenv
```

### `.github/workflows/` — CI/CD Pipelines

| File | Schedule | Purpose |
|------|----------|---------|
| `pipeline.yml`          | Every 2 hours| Fetch → Engineer → Store in MongoDB |
| `train_model.yml`      | Every day | Fetch from MongoDB → Train → Save model |

### `.devcontainer/` — Dev Container
Configuration for running this project in a reproducible containerized development environment (VS Code Dev Containers / GitHub Codespaces compatible).

---

## 🔄 Automated CI/CD

Both pipelines run automatically via GitHub Actions with no manual intervention required.

### Feature Pipeline — Every 2 hours

```yaml
on:
  schedule:
     - cron: "0 */2 * * *"   # every 2 hours
```

1. Calls OpenWeather API for all 7 cities
2. Runs `prepare_features.py` to engineer features
3. Appends new records to MongoDB (deduplication enforced)

### Training Pipeline — Daily

```yaml
on:
  schedule:
    - cron: "0 0 * * *" UTC
```

1. Fetches all feature records from MongoDB
2. Runs `q.py` to retrain models on the latest data
3. Saves the updated model for the dashboard to use

> All API keys and MongoDB credentials are stored securely as **GitHub Secrets**.

---

## 🤖 Models

| Model | Library | Deployed |
|-------|---------|:--------:|
| Random Forest | Scikit-learn | ✅ Yes |
| Ridge Regression | Scikit-learn | ✅ Yes |
| LSTM | TensorFlow         | ⚠️ Local only |

### Why is LSTM not deployed?

The LSTM was trained successfully in the local Anaconda environment and produced strong results. However, when deploying to Streamlit Cloud, the TensorFlow version required by the trained model was incompatible with the Streamlit Cloud environment. Downgrading TensorFlow broke the training code, making it impossible to train and serve in the same environment.

---

##  Acknowledgements

- [OpenWeather API](https://openweathermap.org/api/air-pollution) — real-time air pollution data
- [MongoDB Atlas](https://www.mongodb.com/atlas) — cloud feature store
- [Streamlit](https://streamlit.io/) — dashboard framework
- [SHAP](https://shap.readthedocs.io/) — model explainability
- [Scikit-learn](https://scikit-learn.org/) & [TensorFlow](https://www.tensorflow.org/) — ML modeling

---

<div align="center">
  <strong>Built by Zainub042</strong><br><br>
  <a href="https://aqi-pipeline-crgkkiixn4gzgse8s6j6oy.streamlit.app/">🌐 Live App</a>
</div>
