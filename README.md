
# -Air-Quality-Index-Prediction-System
=======
# India AQI Prediction — Production ML Project

> A production-ready machine learning system for predicting India's National Air Quality Index (NAQI) in real time — with a full demonstration and fix of a data leakage flaw from the GeeksForGeeks tutorial.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Leakage: The GFG Problem](#2-data-leakage-the-gfg-problem)
3. [Soft Leakage Warning](#3-soft-leakage-warning)
4. [India NAQI Formula](#4-india-naqi-formula)
5. [Datasets](#5-datasets)
6. [Project Structure](#6-project-structure)
7. [Quick Start](#7-quick-start)
8. [CLI Reference](#8-cli-reference)
9. [Streamlit App](#9-streamlit-app)
10. [REST API](#10-rest-api)
11. [Testing](#11-testing)
12. [Model Performance](#12-model-performance)
13. [WHO Health Guidelines](#13-who-health-guidelines)
14. [Deployment](#14-deployment)
15. [References](#15-references)

---

## 1. Project Overview

This project builds two complete ML pipelines for Air Quality Index prediction:

| Pipeline | File | Dataset | Best R² | Purpose |
|---|---|---|---|---|
| **GFG (leaky)** | `src/aqi_pipeline.py` | `AQI-and-Lat-Long-of-Countries.csv` | 0.998 | Demonstrates data leakage |
| **Real-time India** | `src/realtime_aqi_pipeline.py` | `Real-time-AQI_data.csv` | 0.97 (pollutants) / 0.52 (geo-only) | Production prediction |

**What this project provides:**
- Real-time AQI prediction for any India location using trained ML models
- 80% prediction confidence intervals via quantile regression
- 7-page Streamlit web app with interactive maps, SHAP explanations, and forecasts
- FastAPI REST endpoint with Swagger UI
- Email alerts when AQI exceeds configurable thresholds
- Model drift monitoring and automated reports
- 80 unit/integration/API tests
- Full data quality validation pipeline

---

## 2. Data Leakage: The GFG Problem

The [GeeksForGeeks AQI tutorial](https://www.geeksforgeeks.org/python/predicting-air-quality-index-using-python/) reports:

```
Mean Absolute Error: 0.09
R² Score: 1.00
```

**These numbers are meaningless.** The India NAQI formula is:

```
AQI = max(sub_index(PM2.5), sub_index(PM10), sub_index(NO2), ...)
```

The tutorial uses the four sub-index columns (CO_AQI, Ozone_AQI, NO2_AQI, PM2.5_AQI) **as input features** to predict `AQI Value`. In 99.8% of rows, `AQI Value` is mathematically the maximum of those columns. The model learns the `max()` function, not anything about air quality.

| Model | R² | What it actually learned |
|---|---|---|
| GFG tutorial (hard leakage) | 0.998 | The `max()` formula |
| Geo-only honest model | 0.523 | Real geographic patterns |

---

## 3. Soft Leakage Warning

The real-time pipeline achieves R²=0.97 when using raw pollutant concentrations (PM2.5, PM10, etc.) as features. This looks impressive but is still a form of **soft leakage**:

- NAQI is a deterministic function of pollutant concentrations
- A model trained on concentrations is approximating the NAQI formula
- It will not generalize to sensors with different calibrations or different pollutant mixes

The geo-only model (R²=0.52) is the honest measure of geographic generalization. Both models are available and clearly labeled in the pipeline output.

---

## 4. India NAQI Formula

India's NAQI uses a piecewise-linear sub-index for each pollutant:

**PM2.5 (µg/m³) breakpoints:**

| Concentration | Sub-index |
|---|---|
| 0 – 30 | 0 – 50 |
| 30 – 60 | 50 – 100 |
| 60 – 90 | 100 – 200 |
| 90 – 120 | 200 – 300 |
| 120 – 250 | 300 – 400 |
| 250+ | 400 – 500 |

**AQI Categories:**

| AQI Range | Category | Color |
|---|---|---|
| 0 – 50 | Good | #00e400 |
| 51 – 100 | Satisfactory | #92d050 |
| 101 – 200 | Moderate | #ffff00 |
| 201 – 300 | Poor | #ff7e00 |
| 301 – 400 | Very Poor | #ff0000 |
| 401 – 500 | Severe | #7e0023 |

Overall NAQI = **max** of all pollutant sub-indices with data.

---

## 5. Datasets

### Primary: Real-time India CPCB Data

**File:** `data/Real-time-AQI_data.csv` (not in git — download separately)
**Source:** India Central Pollution Control Board via OpenAQ
**Format:** Long format — one row per station × pollutant

| Column | Description |
|---|---|
| `country`, `state`, `city`, `station` | Location identifiers |
| `last_update` | Timestamp of measurement |
| `latitude`, `longitude` | Station coordinates |
| `pollutant_id` | One of: CO, NH3, NO2, OZONE, PM10, PM2.5, SO2 |
| `pollutant_avg` | Average concentration (µg/m³ or ppb) |

**After processing:** 487 stations, 7 pollutants, NAQI computed per station.

### Secondary: GFG Global Dataset

**File:** `data/AQI-and-Lat-Long-of-Countries.csv`
**Rows:** 16,695 · **Columns:** 7 · **Purpose:** Leakage demonstration only

---

## 6. Project Structure

```
Predicting Air Quality Index/
├── data/
│   ├── Real-time-AQI_data.csv          # India CPCB data (download manually)
│   ├── AQI-and-Lat-Long-of-Countries.csv  # GFG dataset (download manually)
│   └── historical_aqi.csv              # Collected via fetch_historical.py
│
├── src/
│   ├── config.py                        # Centralized paths, constants, breakpoints
│   ├── realtime_aqi_pipeline.py         # Main pipeline: train, predict, tune, live
│   ├── aqi_pipeline.py                  # GFG leakage demonstration pipeline
│   ├── app.py                           # 7-page Streamlit web app
│   ├── api.py                           # FastAPI REST endpoint
│   ├── data_quality.py                  # Data validation and cleaning
│   ├── forecast.py                      # Hourly/monthly/lag-model forecasting
│   ├── aqi_alert.py                     # Email alert system
│   ├── drift_tracker.py                 # Model drift monitoring
│   ├── fetch_historical.py              # OpenAQ historical data collector
│   └── weather.py                       # OpenWeatherMap integration
│
├── tests/
│   ├── __init__.py
│   ├── test_naqi.py                     # 35 NAQI formula unit tests
│   ├── test_pipeline.py                 # 24 pipeline integration tests
│   └── test_api.py                      # 21 FastAPI endpoint tests
│
├── outputs/
│   ├── figures/                         # All generated PNGs
│   ├── rt_best_model.pkl                # Trained GBM model
│   ├── rt_lower_model.pkl               # Quantile model (10th percentile)
│   ├── rt_upper_model.pkl               # Quantile model (90th percentile)
│   ├── rt_classifier.pkl                # AQI category classifier
│   ├── rt_feature_names.pkl             # Feature names list
│   ├── rt_pollutant_medians.pkl         # Station-level pollutant medians
│   ├── rt_stations_aqi.csv              # Per-station AQI for nearest-neighbor lookup
│   └── rt_shap_explainer.pkl            # SHAP TreeExplainer
│
├── .streamlit/
│   ├── config.toml                      # Theme and server settings
│   └── secrets.toml.example            # Template for API keys
│
├── Dockerfile
├── Makefile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Quick Start

### Prerequisites

```bash
Python >= 3.9
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or using Makefile:
make install
```

### 2. Add your dataset

Place `Real-time-AQI_data.csv` in the `data/` directory.

### 3. Train the pipeline

```bash
python src/realtime_aqi_pipeline.py
# or:
make train
```

This generates all model artifacts in `outputs/` and prints a full training report.

### 4. Launch the web app

```bash
streamlit run src/app.py
# or:
make app
```

---

## 8. CLI Reference

### Main pipeline (`src/realtime_aqi_pipeline.py`)

```bash
# Train all models (GBM, RF, Ridge, LR, DT + quantile models + classifier)
python src/realtime_aqi_pipeline.py

# Train with Optuna hyperparameter tuning (~5 min, 50 trials)
python src/realtime_aqi_pipeline.py --tune

# Predict AQI for a location (shows 80% confidence interval)
python src/realtime_aqi_pipeline.py --predict 28.6139 77.2090

# Fetch live data from OpenAQ and predict
python src/realtime_aqi_pipeline.py --live 28.6139 77.2090
```

### Data quality (`src/data_quality.py`)

```bash
# Run validation report on default dataset
python src/data_quality.py

# Run report on specific file
python src/data_quality.py --file data/Real-time-AQI_data.csv

# Run report and auto-fix issues
python src/data_quality.py --fix
```

### Alerts (`src/aqi_alert.py`)

```bash
# Dry run (no email sent — shows what would be alerted)
python src/aqi_alert.py --dry-run

# Send test email to verify credentials
python src/aqi_alert.py --test-email
```

### Drift monitoring (`src/drift_tracker.py`)

```bash
# Log today's predictions for all 20 cities
python src/drift_tracker.py --log

# Generate drift report (MAE, RMSE, bias, plots)
python src/drift_tracker.py --report

# Log and report in one command (Makefile: make drift)
python src/drift_tracker.py --log --report
```

### Forecast (`src/forecast.py`)

```bash
# Hourly forecast for Delhi with plot
python src/forecast.py --lat 28.6139 --lng 77.2090 --plot
```

### Historical data fetch (`src/fetch_historical.py`)

```bash
# Fetch last 30 days of data for 20 Indian cities from OpenAQ
python src/fetch_historical.py
```

### Makefile shortcuts

```bash
make install    # Install all dependencies
make train      # Train pipeline
make tune       # Train with Optuna (~5 min)
make app        # Launch Streamlit app
make api        # Start FastAPI server (http://localhost:8000/docs)
make test       # Run all 80 tests
make test-naqi  # Run NAQI unit tests only (fastest)
make test-cov   # Run tests with coverage report
make quality    # Run data quality validation
make fetch      # Fetch live OpenAQ data
make alert      # Dry-run email alert check
make drift      # Log predictions + generate drift report
make forecast   # Print hourly forecast for Delhi
make predict    # Predict AQI for Delhi
make clean      # Remove generated model artifacts
make help       # Show all commands
```

---

## 9. Streamlit App

Seven pages accessible from the sidebar:

| Page | Description |
|---|---|
| **Predict for Location** | Enter lat/lng → AQI + 80% CI + pollutant breakdown + WHO comparison + multi-station stats |
| **Forecast** | Hourly (0–23h) and monthly (Jan–Dec) AQI forecasts with plots |
| **City Comparison** | Bar chart of predicted AQI for 20 major Indian cities |
| **Station Map** | Interactive Folium map of all training stations coloured by AQI |
| **SHAP Explanation** | Feature importance bar chart and beeswarm plot |
| **Alerts & Drift** | Configure email alerts; view drift report and rolling MAE chart |
| **About** | Project documentation, leakage explanation, soft leakage note |

### Optional: API keys for enhanced features

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:

```toml
OWM_API_KEY = "your_openweathermap_key"    # Weather context on predict page
ALERT_EMAIL = "you@gmail.com"               # Alert sender
ALERT_PASSWORD = "your_app_password"        # Gmail App Password (not account password)
ALERT_RECIPIENTS = ["you@gmail.com"]
ALERT_THRESHOLD = 200
```

---

## 10. REST API

Start the server:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# or: make api
```

Interactive docs: http://localhost:8000/docs

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root health check |
| `GET` | `/health` | Model load status |
| `GET` | `/predict` | Quick prediction by lat/lng |
| `POST` | `/predict` | Full prediction with pollutant concentrations |
| `GET` | `/predict/interval` | Prediction + 80% confidence interval |
| `GET` | `/cities` | All 20 cities sorted by AQI descending |
| `GET` | `/cities/{name}` | Detailed city prediction + sub-indices + WHO |
| `GET` | `/naqi/compute` | Compute NAQI directly from concentrations (no ML) |

### Example: POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 28.6139,
    "lng": 77.2090,
    "pm25": 145.0,
    "pm10": 210.0,
    "no2": 65.0,
    "so2": 20.0
  }'
```

Response:
```json
{
  "aqi": 187.3,
  "category": "Moderate",
  "lower_80": 142.1,
  "upper_80": 231.8,
  "color": "#ffff00"
}
```

---

## 11. Testing

```bash
# Run all 80 tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run only NAQI formula tests (fastest, no model loading)
pytest tests/test_naqi.py -v
```

### Test coverage

| Test file | Tests | What is covered |
|---|---|---|
| `test_naqi.py` | 35 | NAQI sub-index boundaries, negative/NaN inputs, category mapping, color mapping |
| `test_pipeline.py` | 24 | `load_and_process`, `engineer_features`, `build_input_vector`, `predict_aqi` |
| `test_api.py` | 21 | All API endpoints, validation, error handling, high-pollution assertions |

---

## 12. Model Performance

### Real-time India pipeline (R² on hold-out test set)

| Model | CV R² | Test R² | Test MAE | Notes |
|---|---|---|---|---|
| **Gradient Boosting** | **0.968** | **0.965** | **8.2** | Best model (Optuna-tuned) |
| Random Forest | 0.952 | 0.948 | 9.7 | |
| Decision Tree | 0.891 | 0.884 | 14.1 | |
| Ridge Regression | 0.621 | 0.618 | 28.3 | |
| Linear Regression | 0.618 | 0.615 | 28.5 | |
| **Geo-only (honest)** | **0.523** | **0.510** | **16.4** | No pollutant features |

> **Note:** High R² with pollutant features reflects soft leakage (learning NAQI formula). The geo-only model (0.52) is the honest geographic generalization estimate.

### 80% Prediction Interval

Quantile regression models (alpha=0.10 and alpha=0.90) provide 80% prediction intervals. Coverage on the test set is approximately 80% — calibrated correctly.

### GFG leakage pipeline

| Model | R² | Notes |
|---|---|---|
| Random Forest (leaky) | 0.998 | Using sub-index features = learning max() |
| Random Forest (honest) | 0.510 | Geographic features only |

---

## 13. WHO Health Guidelines

PM2.5 annual mean targets used in the app's WHO comparison table:

| Pollutant | WHO AQG (2021) | India NAQS | Unit |
|---|---|---|---|
| PM2.5 (annual) | 5 | 40 | µg/m³ |
| PM2.5 (24h) | 15 | 60 | µg/m³ |
| PM10 (annual) | 15 | 60 | µg/m³ |
| PM10 (24h) | 45 | 100 | µg/m³ |
| NO2 (annual) | 10 | 40 | µg/m³ |
| SO2 (24h) | 40 | 80 | µg/m³ |
| O3 (8h peak) | 60 | 100 | µg/m³ |

India's National Ambient Air Quality Standards (NAQS) are significantly more permissive than WHO guidelines. Most Indian cities exceed WHO PM2.5 limits throughout the year.

---

## 14. Deployment

### Docker

```bash
# Build image
docker build -t aqi-app .

# Run Streamlit app
docker run -p 8501:8501 aqi-app

# Run API
docker run -p 8000:8000 aqi-app uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Streamlit Cloud

1. Push to GitHub (data files and model artifacts are in `.gitignore` — add them separately or retrain on first run)
2. Connect repo at https://share.streamlit.io
3. Add secrets in the Streamlit Cloud dashboard (same keys as `secrets.toml.example`)

### Environment variables

| Variable | Purpose |
|---|---|
| `OWM_API_KEY` | OpenWeatherMap API key |
| `ALERT_EMAIL` | Gmail sender address |
| `ALERT_PASSWORD` | Gmail App Password |

---

## 15. References

- CPCB India. *National Ambient Air Quality Standards.* Ministry of Environment, Forest and Climate Change, 2009.
- WHO. *WHO Global Air Quality Guidelines.* World Health Organization, 2021.
- Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS, 2017.
- Lundberg, S. & Lee, S.I. *A Unified Approach to Interpreting Model Predictions.* NeurIPS, 2017. (SHAP)
- Akiba, T. et al. *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD, 2019.
- OpenAQ. *Open air quality data platform.* https://openaq.org
- GeeksForGeeks. *Predicting Air Quality Index using Python.* (Original tutorial — contains hard data leakage as documented in Section 2.)
- Breiman, L. *Random Forests.* Machine Learning, 45, 5–32. 2001.

