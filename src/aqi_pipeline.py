"""
AQI Prediction Project — Full Pipeline
=======================================
Dataset : AQI-and-Lat-Long-of-Countries.csv
Purpose : Demonstrate correct ML workflow for AQI regression,
          exposing and fixing the data-leakage issue from the
          GeeksForGeeks tutorial.
"""

import os
import warnings
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
DATA_PATH   = os.path.join(_ROOT, 'data', 'AQI-and-Lat-Long-of-Countries.csv')
OUTPUT_DIR  = os.path.join(_ROOT, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {
    'Good'          : '#2ecc71',
    'Moderate'      : '#f1c40f',
    'Unhealthy'     : '#e67e22',
    'Unhealthy+'    : '#e74c3c',
    'Very Unhealthy': '#8e44ad',
    'Hazardous'     : '#2c3e50',
}
CAT_BINS   = [0, 50, 100, 150, 200, 300, 500]
CAT_LABELS = list(PALETTE.keys())

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
# 1. DATA LOADING & OVERVIEW
# ─────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def overview(df):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {df.shape[1]}")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    print(f"  AQI range   : {df['AQI Value'].min()} – {df['AQI Value'].max()}")
    print(f"  AQI mean    : {df['AQI Value'].mean():.1f}")
    print(f"  AQI median  : {df['AQI Value'].median():.1f}")
    print()
    # Leakage check
    sub_cols = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    computed = df[sub_cols].max(axis=1)
    match_pct = (computed == df['AQI Value']).mean() * 100
    print(f"  ⚠  AQI == max(sub-indices): {match_pct:.1f}% of rows  ← DATA LEAKAGE")
    print()
    return df


# ─────────────────────────────────────────────
# 2. EDA PLOTS
# ─────────────────────────────────────────────
def plot_eda(df):
    sub_cols = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']

    # --- 2a. AQI Category Distribution ---
    cats = pd.cut(df['AQI Value'], bins=CAT_BINS, labels=CAT_LABELS)
    counts = cats.value_counts().reindex(CAT_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('AQI Distribution Analysis', fontsize=15, fontweight='bold', y=1.01)

    bars = axes[0].bar(counts.index, counts.values,
                       color=[PALETTE[c] for c in counts.index],
                       edgecolor='white', linewidth=0.8)
    axes[0].set_title('AQI Category Distribution')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                     f'{val:,}', ha='center', va='bottom', fontsize=9)

    axes[1].hist(df['AQI Value'], bins=60, color='#3498db', edgecolor='white', alpha=0.85)
    axes[1].axvline(df['AQI Value'].mean(),   color='#e74c3c', lw=2, ls='--', label=f"Mean={df['AQI Value'].mean():.0f}")
    axes[1].axvline(df['AQI Value'].median(), color='#2ecc71', lw=2, ls='--', label=f"Median={df['AQI Value'].median():.0f}")
    axes[1].set_title('AQI Value Histogram')
    axes[1].set_xlabel('AQI Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_aqi_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig1_aqi_distribution.png")

    # --- 2b. Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[sub_cols + ['AQI Value', 'lat', 'lng']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, linewidths=0.5, square=True,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig2_correlation_heatmap.png")

    # --- 2c. Pollutant sub-index distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pollutant Sub-Index Distributions', fontsize=14, fontweight='bold')
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for ax, col, color in zip(axes.flat, sub_cols, colors):
        ax.hist(df[col], bins=50, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(df[col].median(), color='black', lw=1.5, ls='--',
                   label=f'Median={df[col].median():.0f}')
        ax.set_title(col)
        ax.set_xlabel('AQI Sub-Index Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_pollutant_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig3_pollutant_distributions.png")

    # --- 2d. Dominant pollutant pie ---
    dominant = df[sub_cols].idxmax(axis=1).value_counts()
    labels_clean = [c.replace(' AQI Value', '') for c in dominant.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    wedge_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12'][:len(dominant)]
    wedges, texts, autotexts = ax.pie(
        dominant.values, labels=labels_clean,
        autopct='%1.1f%%', colors=wedge_colors,
        startangle=140, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax.set_title('Which Pollutant Drives AQI Most Often?', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_dominant_pollutant.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig4_dominant_pollutant.png")

    # --- 2e. Geographic scatter ---
    fig, ax = plt.subplots(figsize=(14, 7))
    scatter = ax.scatter(df['lng'], df['lat'],
                         c=df['AQI Value'], cmap='RdYlGn_r',
                         s=6, alpha=0.5, linewidths=0)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('AQI Value', fontsize=10)
    ax.set_title('Global AQI Distribution by Location', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_geographic_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig5_geographic_scatter.png")


# ─────────────────────────────────────────────
# 3. LEAKAGE DEMONSTRATION
# ─────────────────────────────────────────────
def demonstrate_leakage(df):
    print("=" * 60)
    print("LEAKAGE DEMONSTRATION")
    print("=" * 60)
    sub_cols = ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
    y = df['AQI Value']

    # Leaky model (GFG approach)
    X_leak = df[sub_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(X_leak, y, test_size=0.2, random_state=42)
    rf_leak = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_leak.fit(X_tr, y_tr)
    y_pred_leak = rf_leak.predict(X_te)

    mae_l  = mean_absolute_error(y_te, y_pred_leak)
    rmse_l = np.sqrt(mean_squared_error(y_te, y_pred_leak))
    r2_l   = r2_score(y_te, y_pred_leak)

    print(f"  Leaky model (sub-AQI features):")
    print(f"    MAE  = {mae_l:.4f}   ← suspiciously perfect")
    print(f"    RMSE = {rmse_l:.4f}")
    print(f"    R²   = {r2_l:.6f}  ← data leakage, not real learning")
    print()
    return r2_l


# ─────────────────────────────────────────────
# 4. HONEST MODEL — LAT/LNG
# ─────────────────────────────────────────────
def train_honest_model(df):
    print("=" * 60)
    print("HONEST MODEL (Lat/Lng only — no leakage)")
    print("=" * 60)
    y = df['AQI Value']
    X = df[['lat', 'lng']]

    # Add geographic features
    X = X.copy()
    X['lat_abs']    = np.abs(X['lat'])              # distance from equator
    X['lat_sin']    = np.sin(np.radians(X['lat']))
    X['lng_cos']    = np.cos(np.radians(X['lng']))
    X['lng_sin']    = np.sin(np.radians(X['lng']))
    X['equatorial'] = (np.abs(X['lat']) < 23.5).astype(int)  # tropics flag

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression'  : LinearRegression(),
        'Ridge Regression'   : Ridge(alpha=10),
        'Decision Tree'      : DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest'      : RandomForestRegressor(n_estimators=200, max_depth=15,
                                                     random_state=42, n_jobs=-1),
        'Gradient Boosting'  : GradientBoostingRegressor(n_estimators=200,
                                                         learning_rate=0.1,
                                                         max_depth=5,
                                                         random_state=42),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
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
    print()

    # Best model (already fitted in the loop above)
    best_name   = results_df.iloc[0]['Model']
    best_model  = models[best_name]
    y_pred_best = best_model.predict(X_te)

    print(f"  Best model: {best_name}")
    print(f"    Test R²   = {r2_score(y_te, y_pred_best):.4f}")
    print(f"    Test MAE  = {mean_absolute_error(y_te, y_pred_best):.2f}")
    print(f"    Test RMSE = {np.sqrt(mean_squared_error(y_te, y_pred_best)):.2f}")

    # Save model
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, 'best_model.pkl'))
    print(f"  Model saved to outputs/best_model.pkl")

    return results_df, best_model, X_te, y_te, y_pred_best, X


# ─────────────────────────────────────────────
# 5. MODEL COMPARISON PLOTS
# ─────────────────────────────────────────────
def plot_model_comparison(results_df, y_te, y_pred_best, leaky_r2):
    # --- 5a. Model comparison bar ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Model Comparison (Honest Models — No Leakage)', fontsize=14, fontweight='bold')

    bar_colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]

    for ax, metric in zip(axes, ['CV R² Mean', 'CV MAE Mean', 'Test RMSE']):
        vals = results_df.set_index('Model')[metric]
        bars = ax.barh(vals.index, vals.values, color=bar_colors, edgecolor='white')
        ax.set_title(metric)
        ax.set_xlabel(metric)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig6_model_comparison.png")

    # --- 5b. Actual vs Predicted ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Best Honest Model: Actual vs Predicted', fontsize=13, fontweight='bold')

    # Scatter
    axes[0].scatter(y_te, y_pred_best, alpha=0.3, s=8, color='#3498db')
    mn, mx = y_te.min(), y_te.max()
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual AQI')
    axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title(f'Scatter  (R²={r2_score(y_te, y_pred_best):.3f})')
    axes[0].legend()

    # Residuals
    residuals = y_te - y_pred_best
    axes[1].scatter(y_pred_best, residuals, alpha=0.3, s=8, color='#e74c3c')
    axes[1].axhline(0, color='black', lw=1.5, ls='--')
    axes[1].set_xlabel('Predicted AQI')
    axes[1].set_ylabel('Residual (Actual − Predicted)')
    axes[1].set_title(f'Residuals  (MAE={mean_absolute_error(y_te, y_pred_best):.1f})')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig7_actual_vs_predicted.png")

    # --- 5c. Leaky vs Honest side-by-side ---
    honest_r2 = r2_score(y_te, y_pred_best)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['GFG Model\n(leaky sub-indices)', 'Honest Model\n(geographic features)'],
                  [leaky_r2, honest_r2],
                  color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('R² Score')
    ax.set_title('Leaky vs Honest Model: The Real Difference', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, [leaky_r2, honest_r2]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'R² = {val:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.text(0, leaky_r2 / 2, '⚠ Memorised\nformula', ha='center', color='white',
            fontsize=10, fontweight='bold')
    ax.text(1, honest_r2 / 2, '✓ Real\nlearning', ha='center', color='white',
            fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig8_leaky_vs_honest.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig8_leaky_vs_honest.png")


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (if tree model)
# ─────────────────────────────────────────────
def plot_feature_importance(best_model, X):
    if not hasattr(best_model, 'feature_importances_'):
        return
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#3498db' if v < imp.max() else '#e74c3c' for v in imp.values]
    imp.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Feature Importances (Best Model)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig9_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig9_feature_importance.png")


# ─────────────────────────────────────────────
# 7. AQI CATEGORY CLASSIFICATION
# ─────────────────────────────────────────────
def aqi_to_category(aqi):
    if aqi <= 50:   return 'Good'
    if aqi <= 100:  return 'Moderate'
    if aqi <= 150:  return 'Unhealthy'
    if aqi <= 200:  return 'Unhealthy+'
    if aqi <= 300:  return 'Very Unhealthy'
    return 'Hazardous'


def evaluate_classification(y_te, y_pred_best):
    from sklearn.metrics import classification_report, confusion_matrix

    y_true_cat = [aqi_to_category(v) for v in y_te]
    y_pred_cat = [aqi_to_category(v) for v in y_pred_best]

    print("=" * 60)
    print("CLASSIFICATION ACCURACY (AQI → Category)")
    print("=" * 60)
    print(classification_report(y_true_cat, y_pred_cat, zero_division=0))

    # Confusion matrix heatmap
    labels_present = sorted(set(y_true_cat + y_pred_cat),
                             key=lambda x: CAT_LABELS.index(x) if x in CAT_LABELS else 99)
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels_present)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_present, yticklabels=labels_present,
                linewidths=0.5, ax=ax)
    ax.set_title('Confusion Matrix — AQI Category Prediction', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig10_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/fig10_confusion_matrix.png")


# ─────────────────────────────────────────────
# 8. EXPORT RESULTS TABLE
# ─────────────────────────────────────────────
def export_results(results_df):
    path = os.path.join(OUTPUT_DIR, 'model_results.csv')
    results_df.to_csv(path, index=False)
    print(f"  Saved: model_results.csv")
    print()
    print(results_df.to_string(index=False))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  AQI PREDICTION — FULL PIPELINE")
    print("=" * 60 + "\n")

    print("[1] Loading data...")
    df = load_data(DATA_PATH)
    overview(df)

    print("[2] Running EDA plots...")
    plot_eda(df)
    print()

    print("[3] Demonstrating data leakage...")
    leaky_r2 = demonstrate_leakage(df)
    print()

    print("[4] Training honest models...")
    results_df, best_model, X_te, y_te, y_pred_best, X_feat = train_honest_model(df)
    print()

    print("[5] Generating comparison plots...")
    plot_model_comparison(results_df, y_te, y_pred_best, leaky_r2)
    print()

    print("[6] Feature importance...")
    plot_feature_importance(best_model, X_feat)
    print()

    print("[7] Category classification evaluation...")
    evaluate_classification(y_te, y_pred_best)
    print()

    print("[8] Exporting results...")
    export_results(results_df)
    print()

    print("=" * 60)
    print("  ALL DONE — check outputs/ folder")
    print("=" * 60 + "\n")
