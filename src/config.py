"""
Central configuration for the AQI Prediction project.
Import this instead of hardcoding paths and constants everywhere.
"""

import os

# ── Directory roots ───────────────────────────────────────────
_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR    = os.path.join(_ROOT, 'data')
OUTPUT_DIR  = os.path.join(_ROOT, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
SRC_DIR     = os.path.join(_ROOT, 'src')
TESTS_DIR   = os.path.join(_ROOT, 'tests')

# ── Data files ────────────────────────────────────────────────
RT_DATA_PATH   = os.path.join(DATA_DIR, 'Real-time-AQI_data.csv')
HIST_DATA_PATH = os.path.join(DATA_DIR, 'historical_aqi.csv')
ORIG_DATA_PATH = os.path.join(DATA_DIR, 'AQI-and-Lat-Long-of-Countries.csv')

# ── Model artifacts ───────────────────────────────────────────
RT_MODEL_PATH       = os.path.join(OUTPUT_DIR, 'rt_best_model.pkl')
RT_LOWER_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rt_lower_model.pkl')
RT_UPPER_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rt_upper_model.pkl')
RT_CLF_PATH         = os.path.join(OUTPUT_DIR, 'rt_classifier.pkl')
RT_SHAP_PATH        = os.path.join(OUTPUT_DIR, 'rt_shap_explainer.pkl')
RT_FEAT_PATH        = os.path.join(OUTPUT_DIR, 'rt_feature_names.pkl')
RT_MEDIANS_PATH     = os.path.join(OUTPUT_DIR, 'rt_pollutant_medians.pkl')
RT_STATIONS_PATH    = os.path.join(OUTPUT_DIR, 'rt_stations_aqi.csv')
RT_RESULTS_PATH     = os.path.join(OUTPUT_DIR, 'rt_model_results.csv')
DRIFT_LOG_PATH      = os.path.join(OUTPUT_DIR, 'drift_log.csv')
ALERT_LOG_PATH      = os.path.join(OUTPUT_DIR, 'alert_log.csv')

# ── Training hyperparameters ──────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
CV_FOLDS        = 5
CI_LOWER_ALPHA  = 0.10   # 80% prediction interval lower bound
CI_UPPER_ALPHA  = 0.90   # 80% prediction interval upper bound
OPTUNA_TRIALS   = 50

# ── Pollutants ────────────────────────────────────────────────
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'OZONE', 'NH3', 'CO']

# ── NAQI breakpoints (India) ──────────────────────────────────
NAQI_BREAKPOINTS = {
    'PM2.5': {'c_breaks': [0, 30,  60,  90,  120,  250,  500],
              'i_breaks': [0, 50, 100, 200,  300,  400,  500]},
    'PM10' : {'c_breaks': [0,  50, 100, 250,  350,  430,  600],
              'i_breaks': [0,  50, 100, 200,  300,  400,  500]},
    'NO2'  : {'c_breaks': [0,  40,  80, 180,  280,  400,  600],
              'i_breaks': [0,  50, 100, 200,  300,  400,  500]},
    'SO2'  : {'c_breaks': [0,  40,  80, 380,  800, 1600, 2100],
              'i_breaks': [0,  50, 100, 200,  300,  400,  500]},
    'OZONE': {'c_breaks': [0,  50, 100, 168,  208,  748, 1000],
              'i_breaks': [0,  50, 100, 200,  300,  400,  500]},
    'NH3'  : {'c_breaks': [0, 200, 400, 800, 1200, 1800, 2400],
              'i_breaks': [0,  50, 100, 200,  300,  400,  500]},
    'CO'   : {'c_breaks': [0, 1000, 2000, 10000, 17000, 34000, 50000],
              'i_breaks': [0,   50,  100,   200,   300,   400,   500]},
}

AQI_CATEGORIES = [
    (0,   50,  'Good',         '#2ecc71'),
    (51,  100, 'Satisfactory', '#a8d240'),
    (101, 200, 'Moderate',     '#f1c40f'),
    (201, 300, 'Poor',         '#e67e22'),
    (301, 400, 'Very Poor',    '#e74c3c'),
    (401, 500, 'Severe',       '#8e44ad'),
]

# ── WHO 2021 air quality guidelines (annual mean, µg/m³) ──────
WHO_GUIDELINES = {
    'PM2.5': 5,    # µg/m³ annual mean
    'PM10' : 15,   # µg/m³ annual mean
    'NO2'  : 10,   # µg/m³ annual mean
    'SO2'  : 40,   # µg/m³ 24-hour mean
    'OZONE': 60,   # µg/m³ peak season 8-hour mean
    'CO'   : None, # WHO uses 8-hour mean, not annual
}

# ── Data quality thresholds (per pollutant, µg/m³) ───────────
DQ_MAX_PLAUSIBLE = {
    'PM2.5': 1000,
    'PM10' : 2000,
    'NO2'  : 2000,
    'SO2'  : 5000,
    'OZONE': 2000,
    'NH3'  : 5000,
    'CO'   : 100000,
}

# ── API ───────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Top Indian cities ─────────────────────────────────────────
INDIA_CITIES = [
    ("Delhi",          28.6139, 77.2090),
    ("Mumbai",         19.0760, 72.8777),
    ("Kolkata",        22.5726, 88.3639),
    ("Chennai",        13.0827, 80.2707),
    ("Bangalore",      12.9716, 77.5946),
    ("Hyderabad",      17.3850, 78.4867),
    ("Ahmedabad",      23.0225, 72.5714),
    ("Pune",           18.5204, 73.8567),
    ("Jaipur",         26.9124, 75.7873),
    ("Lucknow",        26.8467, 80.9462),
    ("Kanpur",         26.4499, 80.3319),
    ("Nagpur",         21.1458, 79.0882),
    ("Patna",          25.5941, 85.1376),
    ("Indore",         22.7196, 75.8577),
    ("Bhopal",         23.2599, 77.4126),
    ("Chandigarh",     30.7333, 76.7794),
    ("Surat",          21.1702, 72.8311),
    ("Coimbatore",     11.0168, 76.9558),
    ("Visakhapatnam",  17.6868, 83.2185),
    ("Amritsar",       31.6340, 74.8723),
]
