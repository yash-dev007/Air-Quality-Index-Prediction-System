"""
Real-Time AQI Prediction Pipeline
===================================
Dataset : Real-time-AQI_data.csv  (India CPCB stations, long format)
         + historical_aqi.csv     (optional, built by fetch_historical.py)
AQI     : Computed using India National AQI (NAQI) breakpoints
Features: Geographic + temporal + pollutant concentrations (µg/m³)
Extras  : Optuna hyperparameter tuning, SHAP explainability,
          multi-city comparison plot, OpenAQ live prediction

Run from project root:
    python src/realtime_aqi_pipeline.py              # full training pipeline
    python src/realtime_aqi_pipeline.py --tune       # + Optuna tuning (slow)
    python src/realtime_aqi_pipeline.py --predict 28.6 77.2
    python src/realtime_aqi_pipeline.py --live 28.6 77.2
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
_ROOT       = os.path.join(os.path.dirname(__file__), '..')
DATA_PATH   = os.path.join(_ROOT, 'data', 'Real-time-AQI_data.csv')
HIST_PATH   = os.path.join(_ROOT, 'data', 'historical_aqi.csv')
OUTPUT_DIR  = os.path.join(_ROOT, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# India NAQI breakpoints — all in µg/m³
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

POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'OZONE', 'NH3', 'CO']

# Top Indian cities for multi-city comparison
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

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#f8f9fa',
    'axes.grid'        : True,
    'grid.alpha'       : 0.4,
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


# ─────────────────────────────────────────────
# NAQI COMPUTATION
# ─────────────────────────────────────────────
def compute_sub_index(concentration, pollutant):
    if pd.isna(concentration) or concentration < 0:
        return np.nan
    bp = NAQI_BREAKPOINTS.get(pollutant)
    if bp is None:
        return np.nan
    c_breaks, i_breaks = bp['c_breaks'], bp['i_breaks']
    if concentration >= c_breaks[-1]:
        return 500.0
    for i in range(len(c_breaks) - 1):
        if c_breaks[i] <= concentration <= c_breaks[i + 1]:
            c_lo, c_hi = c_breaks[i], c_breaks[i + 1]
            i_lo, i_hi = i_breaks[i], i_breaks[i + 1]
            return round((i_hi - i_lo) / (c_hi - c_lo) * (concentration - c_lo) + i_lo, 1)
    return np.nan


def aqi_to_category(aqi):
    if pd.isna(aqi):
        return 'Unknown'
    for lo, hi, label, _ in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return 'Severe'


def aqi_to_color(aqi):
    if pd.isna(aqi):
        return '#999999'
    for lo, hi, _, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return color
    return '#8e44ad'


# ─────────────────────────────────────────────
# 1. DATA LOADING & PROCESSING
# ─────────────────────────────────────────────
def load_and_process(path):
    """
    Load long-format CPCB data, pivot to wide, compute India NAQI.
    Also merges historical data if available.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df[df['pollutant_id'].isin(POLLUTANTS)].copy()

    wide = df.pivot_table(
        index=['country', 'state', 'city', 'station', 'last_update', 'latitude', 'longitude'],
        columns='pollutant_id',
        values='pollutant_avg',
        aggfunc='mean',
    ).reset_index()
    wide.columns.name = None

    for p in POLLUTANTS:
        if p not in wide.columns:
            wide[p] = np.nan

    # Compute NAQI sub-indices and overall AQI
    for p in POLLUTANTS:
        wide[f'{p}_idx'] = wide[p].apply(lambda x: compute_sub_index(x, p))
    idx_cols = [f'{p}_idx' for p in POLLUTANTS]
    wide['AQI'] = wide[idx_cols].max(axis=1)
    wide['AQI_Category'] = wide['AQI'].apply(aqi_to_category)
    wide = wide.rename(columns={'latitude': 'lat', 'longitude': 'lng'})
    wide = wide.dropna(subset=['AQI', 'lat', 'lng']).reset_index(drop=True)

    # Merge historical data if it exists
    if os.path.exists(HIST_PATH):
        hist = pd.read_csv(HIST_PATH)
        hist = hist.dropna(subset=['lat', 'lng']).copy()
        for p in POLLUTANTS:
            if p not in hist.columns:
                hist[p] = np.nan
            for pp in POLLUTANTS:
                hist[f'{pp}_idx'] = hist[pp].apply(lambda x: compute_sub_index(x, pp))
        h_idx = [f'{p}_idx' for p in POLLUTANTS]
        hist['AQI'] = hist[h_idx].max(axis=1)
        hist['AQI_Category'] = hist['AQI'].apply(aqi_to_category)
        hist['last_update'] = hist.get('timestamp', pd.NaT)
        wide = pd.concat([wide, hist], ignore_index=True)
        wide = wide.dropna(subset=['AQI']).reset_index(drop=True)
        print(f"  Merged {len(hist)} historical rows.")

    return wide


def overview(df):
    print("=" * 60)
    print("DATASET OVERVIEW (Real-Time India CPCB)")
    print("=" * 60)
    print(f"  Stations    : {len(df):,}")
    print(f"  States      : {df['state'].nunique() if 'state' in df else 'N/A'}")
    print(f"  Cities      : {df['city'].nunique() if 'city' in df else 'N/A'}")
    print(f"  AQI range   : {df['AQI'].min():.0f} – {df['AQI'].max():.0f}")
    print(f"  AQI mean    : {df['AQI'].mean():.1f}")
    print(f"  AQI median  : {df['AQI'].median():.1f}")
    print()
    cats_ordered = [c for _, _, c, _ in AQI_CATEGORIES]
    print("  Category distribution:")
    cats = df['AQI_Category'].value_counts().reindex(cats_ordered).dropna()
    for cat, n in cats.items():
        print(f"    {cat:<20} {int(n):>4}  ({n/len(df)*100:.1f}%)")
    print()
    print("  Pollutant availability:")
    for p in POLLUTANTS:
        n = df[p].notna().sum() if p in df else 0
        print(f"    {p:<8}  {n:>4} / {len(df)}")
    print()


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (geo + temporal + pollutants)
# ─────────────────────────────────────────────
def engineer_features(df):
    """
    Build feature matrix:
      - Geographic: lat, lng, lat_abs, lat_sin, lng_cos, lng_sin
      - Temporal  : hour, day_of_week, month, is_weekend, season
      - Pollutants: 7 concentration columns (median-imputed)
    """
    X = pd.DataFrame()

    # Geographic
    X['lat']     = df['lat']
    X['lng']     = df['lng']
    X['lat_abs'] = np.abs(df['lat'])
    X['lat_sin'] = np.sin(np.radians(df['lat']))
    X['lng_cos'] = np.cos(np.radians(df['lng']))
    X['lng_sin'] = np.sin(np.radians(df['lng']))

    # Temporal — parse 'last_update' if present
    if 'last_update' in df.columns:
        ts = pd.to_datetime(df['last_update'], dayfirst=True, errors='coerce')
        X['hour']       = ts.dt.hour.fillna(12).astype(int)
        X['day_of_week'] = ts.dt.dayofweek.fillna(0).astype(int)
        X['month']      = ts.dt.month.fillna(1).astype(int)
        X['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
        # Season: 1=Winter(Dec-Feb), 2=Spring(Mar-May), 3=Summer(Jun-Aug), 4=Autumn(Sep-Nov)
        X['season']     = ts.dt.month.map(
            lambda m: 1 if m in (12, 1, 2) else 2 if m in (3, 4, 5)
            else 3 if m in (6, 7, 8) else 4
        ).fillna(2).astype(int)
    else:
        X['hour']        = 12
        X['day_of_week'] = 0
        X['month']       = 3
        X['is_weekend']  = 0
        X['season']      = 2

    # Pollutant concentrations (fill missing with column median)
    for p in POLLUTANTS:
        col = df[p].copy() if p in df else pd.Series(np.nan, index=df.index)
        X[p] = col.fillna(col.median() if col.notna().any() else 0.0)

    return X


# ─────────────────────────────────────────────
# 3. EDA PLOTS
# ─────────────────────────────────────────────
def plot_eda(df):
    cat_colors = {cat: color for _, _, cat, color in AQI_CATEGORIES}
    cats_ordered = [c for _, _, c, _ in AQI_CATEGORIES]

    # --- fig1: AQI Distribution ---
    counts = df['AQI_Category'].value_counts().reindex(cats_ordered).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('India Real-Time AQI Distribution', fontsize=15, fontweight='bold')
    bars = axes[0].bar(counts.index, counts.values,
                       color=[cat_colors.get(c, '#999') for c in counts.index],
                       edgecolor='white', linewidth=0.8)
    axes[0].set_title('AQI Category Distribution')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('No. of Stations')
    axes[0].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(int(val)), ha='center', va='bottom', fontsize=9)
    axes[1].hist(df['AQI'], bins=40, color='#3498db', edgecolor='white', alpha=0.85)
    axes[1].axvline(df['AQI'].mean(),   color='#e74c3c', lw=2, ls='--',
                    label=f"Mean={df['AQI'].mean():.0f}")
    axes[1].axvline(df['AQI'].median(), color='#2ecc71', lw=2, ls='--',
                    label=f"Median={df['AQI'].median():.0f}")
    for b in [50, 100, 200, 300, 400]:
        axes[1].axvline(b, color='gray', lw=0.8, ls=':', alpha=0.7)
    axes[1].set_title('AQI Histogram')
    axes[1].set_xlabel('AQI Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig1_aqi_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rt_fig1_aqi_distribution.png")

    # --- fig2: State-wise AQI ---
    if 'state' in df.columns:
        state_aqi = df.groupby('state')['AQI'].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [aqi_to_color(v) for v in state_aqi.values]
        bars = ax.barh(state_aqi.index[::-1], state_aqi.values[::-1],
                       color=colors[::-1], edgecolor='white')
        ax.set_title('Top 15 States by Average AQI', fontsize=13, fontweight='bold')
        ax.set_xlabel('Average AQI')
        for bar, val in zip(bars, state_aqi.values[::-1]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig2_state_aqi.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/rt_fig2_state_aqi.png")

    # --- fig3: Pollutant distributions ---
    available = [p for p in POLLUTANTS if p in df and df[p].notna().sum() > 10]
    n = len(available)
    cols_l = 4
    rows_l = (n + cols_l - 1) // cols_l
    fig, axes = plt.subplots(rows_l, cols_l, figsize=(14, rows_l * 3))
    fig.suptitle('Pollutant Concentration Distributions (µg/m³)', fontsize=14, fontweight='bold')
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    ax_list = axes.flat if rows_l > 1 else axes
    for ax, p, color in zip(ax_list, available, palette):
        data = df[p].dropna()
        ax.hist(data, bins=30, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(data.median(), color='black', lw=1.5, ls='--', label=f'Median={data.median():.0f}')
        ax.set_title(p); ax.set_xlabel('µg/m³'); ax.set_ylabel('Count'); ax.legend(fontsize=8)
    for ax in list(ax_list)[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig3_pollutant_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rt_fig3_pollutant_distributions.png")

    # --- fig4: Geographic scatter ---
    fig, ax = plt.subplots(figsize=(9, 10))
    sc = ax.scatter(df['lng'], df['lat'], c=df['AQI'], cmap='RdYlGn_r',
                    vmin=0, vmax=400, s=40, alpha=0.8, edgecolors='white', linewidths=0.3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('AQI Value', fontsize=10)
    ax.set_title('India Real-Time AQI — Monitoring Stations', fontsize=13, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_xlim(66, 98); ax.set_ylim(6, 36)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig4_geographic_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rt_fig4_geographic_scatter.png")

    # --- fig5: Correlation heatmap ---
    corr_cols = ['AQI', 'lat', 'lng'] + [p for p in POLLUTANTS if p in df and df[p].notna().sum() > 10]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax, linewidths=0.5, square=True, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature–AQI Correlation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig5_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rt_fig5_correlation.png")


# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING (Optuna)
# ─────────────────────────────────────────────
def tune_with_optuna(X_tr, y_tr, n_trials=50):
    """
    Use Optuna to find best GradientBoosting hyperparameters.
    Returns the best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed — skipping tuning.")
        return {}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators' : trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth'    : trial.suggest_int('max_depth', 3, 8),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'subsample'    : trial.suggest_float('subsample', 0.6, 1.0),
            'max_features' : trial.suggest_float('max_features', 0.5, 1.0),
            'random_state' : 42,
        }
        model = GradientBoostingRegressor(**params)
        scores = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best['random_state'] = 42
    print(f"  Best params (R²={study.best_value:.4f}): {best}")
    return best


# ─────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────
def train_models(df, run_tuning=False):
    print("=" * 60)
    print("MODEL TRAINING (Geographic + Temporal + Pollutant Features)")
    print("=" * 60)

    X = engineer_features(df)
    y = df['AQI']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    if run_tuning:
        print(f"  Running Optuna (50 trials) on GradientBoosting ...")
        best_gb_params = tune_with_optuna(X_tr, y_tr, n_trials=50)
    else:
        best_gb_params = {'n_estimators': 200, 'learning_rate': 0.1,
                          'max_depth': 5, 'random_state': 42}

    models = {
        'Linear Regression' : LinearRegression(),
        'Ridge Regression'  : Ridge(alpha=10),
        'Decision Tree'     : DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest'     : RandomForestRegressor(n_estimators=200, max_depth=15,
                                                    random_state=42, n_jobs=-1),
        'Gradient Boosting' : GradientBoostingRegressor(**best_gb_params),
    }

    results = []
    for name, model in models.items():
        cv_r2  = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2', n_jobs=-1)
        cv_mae = cross_val_score(model, X_tr, y_tr, cv=kf,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        results.append({
            'Model'      : name,
            'CV R² Mean' : cv_r2.mean(),
            'CV R² Std'  : cv_r2.std(),
            'CV MAE Mean': -cv_mae.mean(),
            'Test R²'    : r2_score(y_te, y_pred),
            'Test MAE'   : mean_absolute_error(y_te, y_pred),
            'Test RMSE'  : np.sqrt(mean_squared_error(y_te, y_pred)),
        })
        print(f"  {name:<22}  CV R²={cv_r2.mean():.3f}±{cv_r2.std():.3f}  "
              f"Test R²={r2_score(y_te,y_pred):.3f}  MAE={mean_absolute_error(y_te,y_pred):.1f}")

    results_df = pd.DataFrame(results).sort_values('CV R² Mean', ascending=False)
    best_name   = results_df.iloc[0]['Model']
    best_model  = models[best_name]
    y_pred_best = best_model.predict(X_te)

    print(f"\n  Best model : {best_name}")
    print(f"  Test R²    = {r2_score(y_te, y_pred_best):.4f}")
    print(f"  Test MAE   = {mean_absolute_error(y_te, y_pred_best):.2f}")
    print(f"  Test RMSE  = {np.sqrt(mean_squared_error(y_te, y_pred_best)):.2f}")

    # Save pollutant medians (used as fallback when no live readings available)
    pollutant_medians = {p: float(X[p].median()) for p in POLLUTANTS}
    joblib.dump(best_model,        os.path.join(OUTPUT_DIR, 'rt_best_model.pkl'))
    joblib.dump(list(X.columns),   os.path.join(OUTPUT_DIR, 'rt_feature_names.pkl'))
    joblib.dump(pollutant_medians, os.path.join(OUTPUT_DIR, 'rt_pollutant_medians.pkl'))
    print(f"  Model saved -> outputs/rt_best_model.pkl")

    # ── Quantile regression (80% prediction interval) ────────────
    print(f"\n  Training quantile models (80% prediction interval)...")
    lower_gb = GradientBoostingRegressor(loss='quantile', alpha=0.10,
                                          n_estimators=100, max_depth=5, random_state=42)
    upper_gb = GradientBoostingRegressor(loss='quantile', alpha=0.90,
                                          n_estimators=100, max_depth=5, random_state=42)
    lower_gb.fit(X_tr, y_tr)
    upper_gb.fit(X_tr, y_tr)
    lower_pred = lower_gb.predict(X_te)
    upper_pred = upper_gb.predict(X_te)
    coverage = np.mean((y_te.values >= lower_pred) & (y_te.values <= upper_pred))
    avg_width = np.mean(upper_pred - lower_pred)
    print(f"  Interval coverage = {coverage:.1%}  (target: 80%)")
    print(f"  Avg interval width = {avg_width:.1f} AQI units")
    joblib.dump(lower_gb, os.path.join(OUTPUT_DIR, 'rt_lower_model.pkl'))
    joblib.dump(upper_gb, os.path.join(OUTPUT_DIR, 'rt_upper_model.pkl'))
    print(f"  Quantile models saved -> outputs/rt_lower_model.pkl / rt_upper_model.pkl")

    # ── AQI Category Classifier ──────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    cat_order = [c for _, _, c, _ in AQI_CATEGORIES]
    y_cat_tr = y_tr.apply(aqi_to_category)
    y_cat_te = y_te.apply(aqi_to_category)
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_cat_tr)
    y_cat_pred = clf.predict(X_te)
    clf_acc = (y_cat_pred == y_cat_te).mean()
    print(f"\n  AQI Category Classifier (Random Forest):")
    print(f"  Category accuracy = {clf_acc:.3f}")
    print(classification_report(y_cat_te, y_cat_pred,
                                 labels=[c for c in cat_order if c in y_cat_te.unique()],
                                 zero_division=0))
    joblib.dump(clf, os.path.join(OUTPUT_DIR, 'rt_classifier.pkl'))
    print(f"  Classifier saved -> outputs/rt_classifier.pkl")

    return results_df, best_model, X_tr, X_te, y_tr, y_te, y_pred_best, X


# ─────────────────────────────────────────────
# 6. RESULT PLOTS
# ─────────────────────────────────────────────
def plot_results(results_df, y_te, y_pred_best, best_model, X):
    # --- fig6: Model comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Model Comparison — India Real-Time AQI', fontsize=14, fontweight='bold')
    bar_colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
    for ax, metric in zip(axes, ['CV R² Mean', 'CV MAE Mean', 'Test RMSE']):
        vals = results_df.set_index('Model')[metric]
        bars = ax.barh(vals.index, vals.values, color=bar_colors, edgecolor='white')
        ax.set_title(metric); ax.set_xlabel(metric)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig6_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rt_fig6_model_comparison.png")

    # --- fig7: Actual vs Predicted ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Best Model: Actual vs Predicted AQI', fontsize=13, fontweight='bold')
    axes[0].scatter(y_te, y_pred_best, alpha=0.4, s=15, color='#3498db')
    mn, mx = y_te.min(), y_te.max()
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual AQI'); axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title(f'Scatter  (R²={r2_score(y_te, y_pred_best):.3f})'); axes[0].legend()
    residuals = y_te - y_pred_best
    axes[1].scatter(y_pred_best, residuals, alpha=0.4, s=15, color='#e74c3c')
    axes[1].axhline(0, color='black', lw=1.5, ls='--')
    axes[1].set_xlabel('Predicted AQI'); axes[1].set_ylabel('Residual (Actual − Predicted)')
    axes[1].set_title(f'Residuals  (MAE={mean_absolute_error(y_te, y_pred_best):.1f})')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig7_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rt_fig7_actual_vs_predicted.png")

    # --- fig8: Feature importance ---
    if hasattr(best_model, 'feature_importances_'):
        imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#e74c3c' if v == imp.max() else '#3498db' for v in imp.values]
        imp.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
        ax.set_title('Feature Importances (Best Model)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig8_feature_importance.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/rt_fig8_feature_importance.png")


# ─────────────────────────────────────────────
# 7. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
def plot_shap(best_model, X_tr, X_te):
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP plots.")
        return

    print("  Computing SHAP values ...")

    # Use TreeExplainer for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        explainer  = shap.TreeExplainer(best_model)
        shap_vals  = explainer.shap_values(X_te)

        # --- fig9a: SHAP summary (beeswarm) ---
        fig, ax = plt.subplots(figsize=(9, 6))
        shap.summary_plot(shap_vals, X_te, show=False, plot_size=None)
        plt.title('SHAP Feature Impact on AQI Prediction', fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig9a_shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/rt_fig9a_shap_summary.png")

        # --- fig9b: SHAP bar (mean absolute) ---
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_vals, X_te, plot_type='bar', show=False, plot_size=None)
        plt.title('Mean |SHAP| — Global Feature Importance', fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig9b_shap_bar.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: figures/rt_fig9b_shap_bar.png")

        # Save explainer for Streamlit app
        joblib.dump(explainer, os.path.join(OUTPUT_DIR, 'rt_shap_explainer.pkl'))
    else:
        print("  SHAP skipped — model does not support TreeExplainer.")


# ─────────────────────────────────────────────
# 8. MULTI-CITY COMPARISON
# ─────────────────────────────────────────────
def plot_city_comparison(best_model, feature_names):
    """Predict AQI for top Indian cities and visualise as ranked bar chart."""
    rows = []
    for city, lat, lng in INDIA_CITIES:
        X_city = build_input_vector(lat, lng, feature_names=feature_names)
        aqi = float(best_model.predict(X_city)[0])
        aqi = max(0, round(aqi, 1))
        rows.append({'city': city, 'lat': lat, 'lng': lng,
                     'AQI': aqi, 'Category': aqi_to_category(aqi)})

    city_df = pd.DataFrame(rows).sort_values('AQI', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [aqi_to_color(v) for v in city_df['AQI']]
    bars = ax.barh(city_df['city'][::-1], city_df['AQI'][::-1],
                   color=colors[::-1], edgecolor='white', height=0.7)
    for bar, row in zip(bars, city_df.iloc[::-1].itertuples()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{row.AQI:.0f}  {row.Category}', va='center', fontsize=9)
    # Category boundary lines
    for boundary, label in [(50, 'Good'), (100, 'Satisfactory'), (200, 'Moderate'),
                             (300, 'Poor'), (400, 'Very Poor')]:
        ax.axvline(boundary, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlim(0, city_df['AQI'].max() * 1.25)
    ax.set_title('Predicted AQI — Top Indian Cities', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted AQI')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'rt_fig10_city_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/rt_fig10_city_comparison.png")

    # Print table
    print()
    print(f"  {'City':<18} {'AQI':>6}  Category")
    print("  " + "-" * 38)
    for _, row in city_df.iterrows():
        print(f"  {row['city']:<18} {row['AQI']:>6.0f}  {row['Category']}")

    return city_df


# ─────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────
def build_input_vector(lat, lng, pollutants_dict=None, feature_names=None, hour=None, month=None):
    """Build a single-row feature DataFrame matching the training layout."""
    if feature_names is None:
        fp = os.path.join(OUTPUT_DIR, 'rt_feature_names.pkl')
        if not os.path.exists(fp):
            raise FileNotFoundError("Model not trained yet — run the pipeline first.")
        feature_names = joblib.load(fp)

    import datetime
    now = datetime.datetime.now()
    row = {
        'lat'        : lat,
        'lng'        : lng,
        'lat_abs'    : abs(lat),
        'lat_sin'    : np.sin(np.radians(lat)),
        'lng_cos'    : np.cos(np.radians(lng)),
        'lng_sin'    : np.sin(np.radians(lng)),
        'hour'       : hour if hour is not None else now.hour,
        'day_of_week': now.weekday(),
        'month'      : month if month is not None else now.month,
        'is_weekend' : int(now.weekday() >= 5),
        'season'     : (1 if now.month in (12,1,2) else 2 if now.month in (3,4,5)
                        else 3 if now.month in (6,7,8) else 4),
    }
    # Use nearest training station's actual pollutant readings as fallback.
    # This avoids the "all Good" bug from global medians, because the model
    # was trained on correlated lat/lng + pollutant combos — supplying median
    # pollutants at a city coordinate creates an out-of-distribution sample.
    if pollutants_dict is None:
        stations_path = os.path.join(OUTPUT_DIR, 'rt_stations_aqi.csv')
        if os.path.exists(stations_path):
            stations = pd.read_csv(stations_path)
            dists = np.sqrt((stations['lat'] - lat)**2 + (stations['lng'] - lng)**2)
            nearest = stations.iloc[dists.idxmin()]
            pollutants_dict = {p: float(nearest[p]) for p in POLLUTANTS
                               if p in nearest and pd.notna(nearest[p])}
        else:
            medians_path = os.path.join(OUTPUT_DIR, 'rt_pollutant_medians.pkl')
            pollutants_dict = joblib.load(medians_path) if os.path.exists(medians_path) else {}

    for p in POLLUTANTS:
        row[p] = pollutants_dict.get(p, 0.0)

    return pd.DataFrame([row])[feature_names]


def predict_aqi(lat, lng, pollutants_dict=None, hour=None, month=None):
    model = joblib.load(os.path.join(OUTPUT_DIR, 'rt_best_model.pkl'))
    X = build_input_vector(lat, lng, pollutants_dict, hour=hour, month=month)
    aqi = float(model.predict(X)[0])
    aqi = max(0, round(aqi, 1))
    return aqi, aqi_to_category(aqi)


def predict_aqi_with_interval(lat, lng, pollutants_dict=None, hour=None, month=None):
    """
    Predict AQI with an 80% prediction interval from quantile regression.
    Returns (aqi, lower_80, upper_80, category).
    lower_80/upper_80 are None if quantile models haven't been trained yet.
    """
    aqi, cat = predict_aqi(lat, lng, pollutants_dict, hour=hour, month=month)
    lower_path = os.path.join(OUTPUT_DIR, 'rt_lower_model.pkl')
    upper_path = os.path.join(OUTPUT_DIR, 'rt_upper_model.pkl')
    if os.path.exists(lower_path) and os.path.exists(upper_path):
        X = build_input_vector(lat, lng, pollutants_dict, hour=hour, month=month)
        lower = max(0.0, round(float(joblib.load(lower_path).predict(X)[0]), 1))
        upper = max(0.0, round(float(joblib.load(upper_path).predict(X)[0]), 1))
        # Ensure lower <= aqi <= upper (small violations from quantile crossing)
        lower = min(lower, aqi)
        upper = max(upper, aqi)
    else:
        lower = upper = None
    return aqi, lower, upper, cat


def get_city_station_stats(city_lat, city_lng, radius_km=60):
    """
    Return AQI statistics for all monitoring stations within radius_km
    of the given city centre.  Uses Haversine distance.
    Returns dict with n_stations, mean/min/max/std AQI, and per-station list.
    """
    stations_path = os.path.join(OUTPUT_DIR, 'rt_stations_aqi.csv')
    if not os.path.exists(stations_path):
        return None
    stations = pd.read_csv(stations_path)

    # Haversine distance in km
    lat1, lng1 = np.radians(city_lat), np.radians(city_lng)
    lat2 = np.radians(stations['lat'].values)
    lng2 = np.radians(stations['lng'].values)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    dist_km = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    nearby = stations[dist_km <= radius_km].copy()
    if nearby.empty:
        return None

    return {
        'n_stations' : int(len(nearby)),
        'mean_aqi'   : round(float(nearby['AQI'].mean()), 1),
        'min_aqi'    : round(float(nearby['AQI'].min()),  1),
        'max_aqi'    : round(float(nearby['AQI'].max()),  1),
        'std_aqi'    : round(float(nearby['AQI'].std()),  1),
        'stations'   : nearby[['station', 'AQI', 'AQI_Category']].to_dict('records'),
    }


def fetch_openaq_and_predict(lat, lng, radius_km=25):
    """Fetch live data from OpenAQ API, then predict AQI with the trained model."""
    try:
        import requests
    except ImportError:
        print("  Install requests: pip install requests")
        return None

    url = "https://api.openaq.org/v3/locations"
    params = {'coordinates': f'{lat},{lng}', 'radius': radius_km*1000,
              'limit': 5, 'order_by': 'distance'}
    print(f"  Fetching OpenAQ — lat={lat}, lng={lng}, radius={radius_km}km ...")
    resp = requests.get(url, params=params, headers={'Accept': 'application/json'}, timeout=10)
    if resp.status_code != 200:
        print(f"  OpenAQ error: {resp.status_code}"); return None

    results = resp.json().get('results', [])
    if not results:
        print("  No stations found nearby."); return None

    station = results[0]
    print(f"  Nearest station: {station.get('name')}")

    meas_resp = requests.get(f"https://api.openaq.org/v3/locations/{station['id']}/latest",
                             headers={'Accept': 'application/json'}, timeout=10)
    if meas_resp.status_code != 200:
        print(f"  Measurements error: {meas_resp.status_code}"); return None

    param_map = {'pm25':'PM2.5','pm2.5':'PM2.5','pm10':'PM10','no2':'NO2',
                 'so2':'SO2','o3':'OZONE','nh3':'NH3','co':'CO'}
    pollutants_found = {}
    for m in meas_resp.json().get('results', []):
        p = m.get('parameter', '').lower()
        v = m.get('value')
        if p in param_map and v is not None and v >= 0:
            pollutants_found[param_map[p]] = v

    aqi_pred, category = predict_aqi(lat, lng, pollutants_found)
    raw_sub = {p: compute_sub_index(v, p) for p, v in pollutants_found.items()
               if not pd.isna(compute_sub_index(v, p))}

    print()
    print("  ─── REAL-TIME AQI RESULT ─────────────────")
    print(f"  Location     : lat={lat}, lng={lng}")
    print(f"  Station      : {station.get('name')}")
    for p, v in pollutants_found.items():
        sub = f"  sub-idx={raw_sub[p]:.0f}" if p in raw_sub else ""
        print(f"  {p:<8}     : {v:.1f} µg/m³{sub}")
    if raw_sub:
        print(f"  NAQI (direct): {max(raw_sub.values()):.1f} ->{aqi_to_category(max(raw_sub.values()))}")
    print(f"  ML Predicted : {aqi_pred:.1f} ->{category}")
    print("  ─────────────────────────────────────────")
    return aqi_pred, category


# ─────────────────────────────────────────────
# 9. EXPORT
# ─────────────────────────────────────────────
def export_results(df, results_df):
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'rt_model_results.csv'), index=False)
    print(f"  Saved: rt_model_results.csv")

    cols = ['state', 'city', 'station', 'lat', 'lng', 'AQI', 'AQI_Category'] + \
           [p for p in POLLUTANTS if p in df]
    df[cols].to_csv(os.path.join(OUTPUT_DIR, 'rt_stations_aqi.csv'), index=False)
    print(f"  Saved: rt_stations_aqi.csv  ({len(df)} stations)")
    print()
    print(results_df.to_string(index=False))


# ─────────────────────────────────────────────
# SOFT LEAKAGE DEMONSTRATION
# ─────────────────────────────────────────────
def demonstrate_soft_leakage(df):
    """
    Explains why R²=0.99 with pollutant features is a form of soft leakage.

    India NAQI is defined as:
        AQI = max(sub_index(PM2.5), sub_index(PM10), ..., sub_index(CO))

    where each sub_index is a deterministic piecewise-linear function of the
    concentration.  Therefore, the model with pollutant concentrations as
    features is *approximating a deterministic formula*, not genuinely
    generalising.  The honest measure of real-world predictive power is the
    geo-only model, which achieves R²~0.52.

    This is a more subtle version of the original GFG leakage — not as
    egregious (concentrations are legitimate real-world inputs), but the
    high R² does not mean "great model"; it means "learned the formula".

    The two use cases:
      - Geographic forecast  : predict AQI without sensor data -> geo-only model
      - Real-time estimation : have sensor readings, want fast NAQI -> concentration model
    """
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.ensemble import RandomForestRegressor

    print("=" * 60)
    print("SOFT LEAKAGE ANALYSIS")
    print("=" * 60)
    print("  NAQI = max(sub_index(concentration)) is a deterministic formula.")
    print("  A model trained WITH concentrations is learning that formula.")
    print("  R²=0.99 does not mean the model generalises — it means it")
    print("  approximated a known mathematical transformation.")
    print()

    X_full = engineer_features(df)
    y = df['AQI']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Model A: geo + temporal only (no pollutant concentrations)
    geo_cols = ['lat', 'lng', 'lat_abs', 'lat_sin', 'lng_cos', 'lng_sin',
                'hour', 'day_of_week', 'month', 'is_weekend', 'season']
    X_geo = X_full[geo_cols]
    rf_geo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    r2_geo = cross_val_score(rf_geo, X_geo, y, cv=kf, scoring='r2', n_jobs=-1).mean()

    # Model B: all features (geo + pollutants)
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    r2_full = cross_val_score(rf_full, X_full, y, cv=kf, scoring='r2', n_jobs=-1).mean()

    print(f"  Geo + temporal only  : CV R² = {r2_geo:.3f}  <- real predictive power")
    print(f"  + pollutant concentrations: CV R² = {r2_full:.3f}  <- learning the NAQI formula")
    print()
    print(f"  Gap ({r2_full - r2_geo:.3f}) = model is recovering NAQI from its own inputs")
    print()
    print("  Conclusion:")
    print("    Use the GEO model  -> 'Where will air quality be bad? (no sensor needed)'")
    print("    Use the FULL model -> 'I have sensor readings — what is the NAQI?'")
    print("    The full model is useful, but its R²=0.99 is not a 'learning' achievement.")
    print()
    return r2_geo, r2_full


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-Time India AQI Prediction')
    parser.add_argument('--tune',    action='store_true', help='Run Optuna hyperparameter tuning')
    parser.add_argument('--predict', nargs=2, type=float, metavar=('LAT', 'LNG'))
    parser.add_argument('--live',    nargs=2, type=float, metavar=('LAT', 'LNG'))
    args = parser.parse_args()

    if args.predict:
        lat, lng = args.predict
        aqi, lower, upper, cat = predict_aqi_with_interval(lat, lng)
        print(f"\nPredicted AQI for lat={lat}, lng={lng}")
        print(f"  AQI      : {aqi}")
        if lower is not None:
            print(f"  80% CI   : {lower:.0f} - {upper:.0f}")
        print(f"  Category : {cat}\n")

    elif args.live:
        lat, lng = args.live
        fetch_openaq_and_predict(lat, lng)

    else:
        print("\n" + "=" * 60)
        print("  REAL-TIME AQI PREDICTION — FULL PIPELINE")
        print("=" * 60 + "\n")

        print("[1] Loading and processing data...")
        df = load_and_process(DATA_PATH)
        overview(df)

        print("[2] Running EDA plots...")
        plot_eda(df)
        print()

        print("[3] Training models" + (" + Optuna tuning" if args.tune else "") + "...")
        results_df, best_model, X_tr, X_te, y_tr, y_te, y_pred_best, X_feat = \
            train_models(df, run_tuning=args.tune)
        print()

        print("[4] Result plots...")
        plot_results(results_df, y_te, y_pred_best, best_model, X_feat)
        print()

        print("[5] SHAP explainability...")
        plot_shap(best_model, X_feat.iloc[X_tr.index], X_feat.iloc[X_te.index])
        print()

        print("[5b] Soft leakage analysis...")
        demonstrate_soft_leakage(df)

        print("[6] Multi-city comparison...")
        feature_names = list(X_feat.columns)
        plot_city_comparison(best_model, feature_names)
        print()

        print("[7] Exporting results...")
        export_results(df, results_df)
        print()

        print("=" * 60)
        print("  ALL DONE — check outputs/ folder")
        print("=" * 60)
        print()
        print("  Streamlit app:  streamlit run src/app.py")
        print("  Collect data:   python src/fetch_historical.py")
        print("  Predict:        python src/realtime_aqi_pipeline.py --predict 28.6 77.2")
        print("  Live fetch:     python src/realtime_aqi_pipeline.py --live 28.6 77.2")
        print("  Tune:           python src/realtime_aqi_pipeline.py --tune")
        print()
