"""
Model Drift Tracker
=====================
Logs predictions and (when available) actual AQI values.
Computes rolling MAE and R² to detect model degradation over time.

Usage:
    python src/drift_tracker.py --log             # log today's predictions
    python src/drift_tracker.py --report          # print drift report + plot
    python src/drift_tracker.py --log --report    # both
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from realtime_aqi_pipeline import (
    predict_aqi, INDIA_CITIES, OUTPUT_DIR, FIGURES_DIR,
)

DRIFT_LOG = os.path.join(OUTPUT_DIR, 'drift_log.csv')
DRIFT_COLS = ['timestamp', 'city', 'lat', 'lng', 'predicted_aqi',
              'actual_aqi', 'error', 'category_match']


def log_predictions(cities=None):
    """
    Log current predictions for all cities.
    actual_aqi is left blank — fill it in later when real readings are available,
    or use fetch_historical.py to get actuals and merge them via update_actuals().
    """
    cities = cities or INDIA_CITIES
    rows = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    for name, lat, lng in cities:
        aqi, cat = predict_aqi(lat, lng)
        rows.append({
            'timestamp'    : ts,
            'city'         : name,
            'lat'          : lat,
            'lng'          : lng,
            'predicted_aqi': aqi,
            'actual_aqi'   : None,
            'error'        : None,
            'category_match': None,
        })
        print(f"  {name:<18} predicted={aqi:.0f}  {cat}")

    df_new = pd.DataFrame(rows)
    if os.path.exists(DRIFT_LOG):
        df_new = pd.concat([pd.read_csv(DRIFT_LOG), df_new], ignore_index=True)
    df_new.to_csv(DRIFT_LOG, index=False)
    print(f"  Logged {len(rows)} predictions -> outputs/drift_log.csv")
    return df_new


def update_actuals(actuals_dict, timestamp=None):
    """
    Update actual AQI values in the drift log after real readings are known.
    actuals_dict: {'Delhi': 145, 'Mumbai': 88, ...}
    timestamp   : match rows with this timestamp (default: most recent)
    """
    if not os.path.exists(DRIFT_LOG):
        print("No drift log found. Run --log first.")
        return

    df = pd.read_csv(DRIFT_LOG)
    if timestamp is None:
        timestamp = df['timestamp'].max()

    mask = df['timestamp'] == timestamp
    for city, actual in actuals_dict.items():
        city_mask = mask & (df['city'] == city)
        df.loc[city_mask, 'actual_aqi'] = actual
        pred = df.loc[city_mask, 'predicted_aqi']
        if not pred.empty:
            from realtime_aqi_pipeline import aqi_to_category
            df.loc[city_mask, 'error'] = actual - pred.values[0]
            df.loc[city_mask, 'category_match'] = int(
                aqi_to_category(actual) == aqi_to_category(pred.values[0])
            )
    df.to_csv(DRIFT_LOG, index=False)
    print(f"Updated actuals for {len(actuals_dict)} cities at {timestamp}")


def drift_report():
    """Print drift summary and generate drift plot."""
    if not os.path.exists(DRIFT_LOG):
        print("No drift log found. Run --log first.")
        return

    df = pd.read_csv(DRIFT_LOG)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_with_actual = df.dropna(subset=['actual_aqi', 'error'])

    print("=" * 55)
    print("MODEL DRIFT REPORT")
    print("=" * 55)
    print(f"  Total logged predictions : {len(df)}")
    print(f"  Predictions with actuals : {len(df_with_actual)}")
    print(f"  Date range               : {df['timestamp'].min().date()} to "
          f"{df['timestamp'].max().date()}")

    if df_with_actual.empty:
        print("\n  No actuals available yet.")
        print("  Use update_actuals() or wait for historical data to accumulate.")
        _plot_predictions_only(df)
        return

    # Overall metrics
    mae   = df_with_actual['error'].abs().mean()
    rmse  = np.sqrt((df_with_actual['error']**2).mean())
    bias  = df_with_actual['error'].mean()
    cat_acc = df_with_actual['category_match'].mean() if 'category_match' in df_with_actual else None

    print(f"\n  Overall (n={len(df_with_actual)}):")
    print(f"    MAE      = {mae:.2f}")
    print(f"    RMSE     = {rmse:.2f}")
    print(f"    Bias     = {bias:+.2f}  ({'over' if bias < 0 else 'under'}-predicting)")
    if cat_acc is not None:
        print(f"    Cat acc  = {cat_acc:.1%}")

    # Rolling MAE over time
    df_with_actual = df_with_actual.sort_values('timestamp')
    df_with_actual['rolling_mae'] = df_with_actual['error'].abs().rolling(5, min_periods=1).mean()

    # Per-city breakdown
    print(f"\n  Per-city breakdown:")
    city_stats = df_with_actual.groupby('city').agg(
        n=('error', 'count'),
        mae=('error', lambda x: x.abs().mean()),
        bias=('error', 'mean'),
    ).sort_values('mae', ascending=False)
    print(city_stats.to_string())

    _plot_drift(df_with_actual, mae, rmse)


def _plot_predictions_only(df):
    """Plot prediction history when no actuals available."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    for city in df['city'].unique():
        city_df = df[df['city'] == city].sort_values('timestamp')
        ax.plot(city_df['timestamp'], city_df['predicted_aqi'],
                marker='o', ms=4, label=city, alpha=0.7)
    ax.set_title('Prediction History by City', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted AQI')
    ax.set_xlabel('Date')
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'rt_drift_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/rt_drift_predictions.png")


def _plot_drift(df, mae, rmse):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Model Drift Report', fontsize=14, fontweight='bold')

    # Rolling MAE over time
    axes[0].plot(df['timestamp'], df['rolling_mae'], color='#e74c3c', lw=2)
    axes[0].axhline(mae, color='black', lw=1, ls='--', label=f'Overall MAE={mae:.1f}')
    axes[0].set_title('Rolling MAE (window=5)')
    axes[0].set_xlabel('Date'); axes[0].set_ylabel('MAE'); axes[0].legend()

    # Error distribution
    axes[1].hist(df['error'], bins=20, color='#3498db', edgecolor='white', alpha=0.85)
    axes[1].axvline(0, color='black', lw=1.5, ls='--')
    axes[1].axvline(df['error'].mean(), color='#e74c3c', lw=1.5, ls='--',
                    label=f"Bias={df['error'].mean():+.1f}")
    axes[1].set_title('Prediction Error Distribution')
    axes[1].set_xlabel('Error (Actual - Predicted)'); axes[1].legend()

    # Actual vs Predicted scatter
    axes[2].scatter(df['actual_aqi'], df['predicted_aqi'], alpha=0.5, s=20, color='#2ecc71')
    mn = min(df['actual_aqi'].min(), df['predicted_aqi'].min())
    mx = max(df['actual_aqi'].max(), df['predicted_aqi'].max())
    axes[2].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')
    axes[2].set_xlabel('Actual AQI'); axes[2].set_ylabel('Predicted AQI')
    axes[2].set_title(f'Actual vs Predicted (RMSE={rmse:.1f})')
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'rt_drift_report.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: figures/rt_drift_report.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Drift Tracker')
    parser.add_argument('--log',    action='store_true', help='Log predictions for all cities')
    parser.add_argument('--report', action='store_true', help='Print drift report')
    args = parser.parse_args()

    if not args.log and not args.report:
        parser.print_help()
        sys.exit(0)

    if args.log:
        print("Logging predictions...")
        log_predictions()

    if args.report:
        drift_report()
