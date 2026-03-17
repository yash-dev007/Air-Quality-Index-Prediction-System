"""
AQI Forecast Module
====================
Generates hourly (0–23h) and monthly (Jan–Dec) AQI forecasts
for a given location using the trained model's temporal features.

If historical_aqi.csv exists (built by fetch_historical.py),
also trains a lag-feature model for next-day prediction.

Usage:
    python src/forecast.py --lat 28.6 --lng 77.2
    python src/forecast.py --lat 28.6 --lng 77.2 --plot
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from realtime_aqi_pipeline import (
    build_input_vector, predict_aqi, aqi_to_category, aqi_to_color,
    AQI_CATEGORIES, POLLUTANTS, OUTPUT_DIR, FIGURES_DIR,
)

HIST_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_aqi.csv')


# ─────────────────────────────────────────────
# HOURLY FORECAST (0–23)
# ─────────────────────────────────────────────
def hourly_forecast(lat, lng, month=None, pollutants_dict=None):
    """Predict AQI for each hour of the day at a given location."""
    import datetime
    month = month or datetime.datetime.now().month
    model = joblib.load(os.path.join(OUTPUT_DIR, 'rt_best_model.pkl'))
    feature_names = joblib.load(os.path.join(OUTPUT_DIR, 'rt_feature_names.pkl'))

    rows = []
    for h in range(24):
        X = build_input_vector(lat, lng, pollutants_dict, feature_names=feature_names,
                               hour=h, month=month)
        aqi = float(model.predict(X)[0])
        aqi = max(0, round(aqi, 1))
        rows.append({'hour': h, 'AQI': aqi, 'Category': aqi_to_category(aqi)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MONTHLY FORECAST (Jan–Dec)
# ─────────────────────────────────────────────
def monthly_forecast(lat, lng, hour=None, pollutants_dict=None):
    """Predict average AQI for each month at a given location."""
    import datetime
    hour = hour if hour is not None else datetime.datetime.now().hour
    model = joblib.load(os.path.join(OUTPUT_DIR, 'rt_best_model.pkl'))
    feature_names = joblib.load(os.path.join(OUTPUT_DIR, 'rt_feature_names.pkl'))
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']

    rows = []
    for m in range(1, 13):
        X = build_input_vector(lat, lng, pollutants_dict, feature_names=feature_names,
                               hour=hour, month=m)
        aqi = float(model.predict(X)[0])
        aqi = max(0, round(aqi, 1))
        rows.append({'month': m, 'month_name': month_names[m-1],
                     'AQI': aqi, 'Category': aqi_to_category(aqi)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# NEXT-DAY PREDICTION (lag-feature model)
# ─────────────────────────────────────────────
def build_lag_features(hist_df, city=None, lags=(1, 2, 3, 7)):
    """
    Build lag features from historical data for a given city.
    Requires historical_aqi.csv with columns: timestamp, city, AQI.
    """
    df = hist_df.copy()
    if city:
        df = df[df['city'] == city]
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    df = df.dropna(subset=['AQI'])
    if len(df) < max(lags) + 5:
        return None, None

    for lag in lags:
        df[f'AQI_lag{lag}'] = df['AQI'].shift(lag)
    df['AQI_rolling3'] = df['AQI'].shift(1).rolling(3).mean()
    df['AQI_rolling7'] = df['AQI'].shift(1).rolling(7).mean()
    df = df.dropna()

    feat_cols = [f'AQI_lag{l}' for l in lags] + ['AQI_rolling3', 'AQI_rolling7']
    X = df[feat_cols]
    y = df['AQI']
    return X, y


def train_lag_model(city=None):
    """
    Train a next-day AQI predictor using lag features from historical data.
    Returns (model, feature_names, last_row) or None if insufficient data.
    """
    if not os.path.exists(HIST_PATH):
        return None

    hist = pd.read_csv(HIST_PATH)
    hist['timestamp'] = pd.to_datetime(hist['timestamp'], errors='coerce')
    X, y = build_lag_features(hist, city=city)
    if X is None:
        return None

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                shuffle=False)  # time-ordered
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"  Lag model — MAE={mean_absolute_error(y_te,y_pred):.1f}  "
          f"R²={r2_score(y_te,y_pred):.3f}  (n={len(X)} days)")

    last_row = X.iloc[[-1]].copy()
    return model, list(X.columns), last_row


def predict_next_day(city=None):
    """Predict tomorrow's AQI from historical lag features."""
    result = train_lag_model(city)
    if result is None:
        return None, "Insufficient historical data. Run fetch_historical.py daily to build history."
    model, feat_names, last_row = result
    aqi = float(model.predict(last_row)[0])
    aqi = max(0, round(aqi, 1))
    return aqi, aqi_to_category(aqi)


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_hourly(hourly_df, lat, lng, save=True):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = [aqi_to_color(v) for v in hourly_df['AQI']]
    ax.bar(hourly_df['hour'], hourly_df['AQI'], color=colors, edgecolor='white', width=0.8)
    ax.plot(hourly_df['hour'], hourly_df['AQI'], 'k-o', ms=4, lw=1.5, alpha=0.6)
    for bnd in [50, 100, 200, 300]:
        ax.axhline(bnd, color='gray', lw=0.7, ls=':', alpha=0.5)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, fontsize=8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Predicted AQI')
    ax.set_title(f'Hourly AQI Forecast  (lat={lat:.2f}, lng={lng:.2f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'rt_forecast_hourly.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/rt_forecast_hourly.png")
    return fig


def plot_monthly(monthly_df, lat, lng, save=True):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [aqi_to_color(v) for v in monthly_df['AQI']]
    ax.bar(monthly_df['month_name'], monthly_df['AQI'], color=colors,
           edgecolor='white', width=0.7)
    ax.plot(monthly_df['month_name'], monthly_df['AQI'], 'k-o', ms=5, lw=1.5, alpha=0.6)
    for bnd in [50, 100, 200, 300]:
        ax.axhline(bnd, color='gray', lw=0.7, ls=':', alpha=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Predicted AQI')
    ax.set_title(f'Seasonal AQI Forecast  (lat={lat:.2f}, lng={lng:.2f})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, 'rt_forecast_monthly.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: figures/rt_forecast_monthly.png")
    return fig


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AQI Forecast')
    parser.add_argument('--lat',   type=float, default=28.6139)
    parser.add_argument('--lng',   type=float, default=77.2090)
    parser.add_argument('--city',  type=str,   default=None,
                        help='City name for lag model (must exist in historical_aqi.csv)')
    parser.add_argument('--plot',  action='store_true', help='Save forecast plots')
    args = parser.parse_args()

    print(f"\nAQI Forecast for lat={args.lat}, lng={args.lng}\n")

    print("Hourly forecast (today):")
    hourly = hourly_forecast(args.lat, args.lng)
    for _, row in hourly.iterrows():
        print(f"  {int(row.hour):02d}:00  AQI={row.AQI:.0f}  {row.Category}")
    if args.plot:
        plot_hourly(hourly, args.lat, args.lng)

    print("\nMonthly / seasonal forecast:")
    monthly = monthly_forecast(args.lat, args.lng)
    for _, row in monthly.iterrows():
        print(f"  {row.month_name}  AQI={row.AQI:.0f}  {row.Category}")
    if args.plot:
        plot_monthly(monthly, args.lat, args.lng)

    print("\nNext-day prediction (lag model):")
    aqi_nd, cat_nd = predict_next_day(args.city)
    if aqi_nd:
        print(f"  Tomorrow: AQI={aqi_nd:.0f}  {cat_nd}")
    else:
        print(f"  {cat_nd}")
