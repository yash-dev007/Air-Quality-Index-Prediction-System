"""
Data Quality Validation Module
================================
Validates raw CPCB sensor data before training. Detects:
  - Missing values
  - Negative concentrations (sensor error)
  - Implausibly high values (sensor spike / calibration fault)
  - Stuck sensors (constant value repeated many times)
  - Outliers (> 3 standard deviations)
  - Sparse pollutant coverage per station

Usage:
    python src/data_quality.py                        # validate default dataset
    python src/data_quality.py --file data/my.csv     # validate custom file
    python src/data_quality.py --fix                  # also write cleaned CSV
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import (RT_DATA_PATH, FIGURES_DIR, OUTPUT_DIR,
                    POLLUTANTS, DQ_MAX_PLAUSIBLE)

STUCK_THRESHOLD    = 5     # flag if same value appears ≥ N times for one station
OUTLIER_SIGMA      = 3.5   # flag if value > mean + N*std (per pollutant)
MIN_COVERAGE_PCT   = 0.5   # flag station if fewer than 50% of pollutants available


# ─────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────
def check_missing(df):
    total = len(df)
    results = {}
    for col in ['latitude', 'longitude', 'pollutant_avg', 'pollutant_id']:
        n_missing = df[col].isna().sum()
        results[col] = {'n_missing': int(n_missing), 'pct': n_missing / total * 100}
    return results


def check_negatives(df):
    neg = df[df['pollutant_avg'] < 0]
    return {
        'n_negative': len(neg),
        'by_pollutant': neg.groupby('pollutant_id').size().to_dict(),
    }


def check_implausible(df):
    flagged = []
    for p in POLLUTANTS:
        pdata = df[df['pollutant_id'] == p]
        max_ok = DQ_MAX_PLAUSIBLE.get(p)
        if max_ok is None:
            continue
        bad = pdata[pdata['pollutant_avg'] > max_ok]
        if not bad.empty:
            flagged.append({
                'pollutant': p,
                'n_flagged': len(bad),
                'max_seen': float(pdata['pollutant_avg'].max()),
                'threshold': max_ok,
                'stations': bad['station'].unique().tolist()[:5],
            })
    return flagged


def check_outliers(df):
    flagged = []
    for p in POLLUTANTS:
        pdata = df[df['pollutant_id'] == p]['pollutant_avg'].dropna()
        if len(pdata) < 10:
            continue
        mean, std = pdata.mean(), pdata.std()
        cutoff = mean + OUTLIER_SIGMA * std
        n_out  = (pdata > cutoff).sum()
        if n_out > 0:
            flagged.append({
                'pollutant': p,
                'n_outliers': int(n_out),
                'mean': round(mean, 1),
                'std': round(std, 1),
                'cutoff': round(cutoff, 1),
                'max_val': round(float(pdata.max()), 1),
            })
    return flagged


def check_stuck_sensors(df):
    stuck = []
    for station in df['station'].unique():
        sdata = df[df['station'] == station]
        for p in POLLUTANTS:
            pdata = sdata[sdata['pollutant_id'] == p]['pollutant_avg']
            if len(pdata) < STUCK_THRESHOLD:
                continue
            mode_count = pdata.value_counts().max()
            if mode_count >= STUCK_THRESHOLD:
                stuck_val = pdata.value_counts().idxmax()
                stuck.append({
                    'station': station,
                    'pollutant': p,
                    'stuck_value': float(stuck_val),
                    'repeat_count': int(mode_count),
                })
    return stuck


def check_station_coverage(df):
    """Check how many pollutants each station has readings for."""
    station_pollutants = df.groupby('station')['pollutant_id'].nunique()
    total_pollutants   = len(POLLUTANTS)
    sparse = station_pollutants[station_pollutants / total_pollutants < MIN_COVERAGE_PCT]
    return {
        'total_stations': int(station_pollutants.shape[0]),
        'sparse_stations': int(len(sparse)),
        'avg_pollutants_per_station': round(float(station_pollutants.mean()), 1),
        'sparse_names': sparse.index.tolist()[:10],
    }


# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────
def clean_data(df):
    """
    Apply conservative cleaning:
      1. Drop rows with missing lat/lng or pollutant_avg
      2. Clamp negative concentrations to 0
      3. Cap implausibly high values at the plausible maximum
      4. Remove statistical outliers (> OUTLIER_SIGMA * std)
    """
    df = df.copy()

    # 1. Drop missing essentials
    df = df.dropna(subset=['latitude', 'longitude', 'pollutant_avg'])

    # 2. Clamp negatives
    df.loc[df['pollutant_avg'] < 0, 'pollutant_avg'] = 0

    # 3. Cap implausible highs
    for p in POLLUTANTS:
        max_ok = DQ_MAX_PLAUSIBLE.get(p)
        if max_ok:
            mask = (df['pollutant_id'] == p) & (df['pollutant_avg'] > max_ok)
            df.loc[mask, 'pollutant_avg'] = max_ok

    # 4. Remove outliers per pollutant
    rows_before = len(df)
    keep_masks = []
    for p in POLLUTANTS:
        pmask  = df['pollutant_id'] == p
        pdata  = df.loc[pmask, 'pollutant_avg']
        if len(pdata) < 10:
            keep_masks.append(pmask.apply(lambda x: True))
            continue
        cutoff = pdata.mean() + OUTLIER_SIGMA * pdata.std()
        keep   = ~pmask | (df['pollutant_avg'] <= cutoff)
        keep_masks.append(keep)

    combined_keep = keep_masks[0]
    for m in keep_masks[1:]:
        combined_keep = combined_keep & m
    df = df[combined_keep].reset_index(drop=True)

    rows_removed = rows_before - len(df)
    return df, rows_removed


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────
def run_report(df, verbose=True):
    issues = []

    missing = check_missing(df)
    neg     = check_negatives(df)
    impl    = check_implausible(df)
    outl    = check_outliers(df)
    stuck   = check_stuck_sensors(df)
    cov     = check_station_coverage(df)

    if verbose:
        print("=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"  Total rows      : {len(df):,}")
        print(f"  Stations        : {df['station'].nunique():,}")
        print(f"  Pollutants      : {df['pollutant_id'].nunique()}")
        print()

        print("  Missing values:")
        for col, info in missing.items():
            flag = " [ISSUE]" if info['n_missing'] > 0 else ""
            print(f"    {col:<20} {info['n_missing']:>5} ({info['pct']:.1f}%){flag}")
            if info['n_missing'] > 0:
                issues.append(f"Missing values in {col}: {info['n_missing']}")

        print(f"\n  Negative concentrations: {neg['n_negative']}")
        if neg['n_negative'] > 0:
            print(f"    By pollutant: {neg['by_pollutant']}")
            issues.append(f"Negative concentrations: {neg['n_negative']} rows")

        if impl:
            print(f"\n  Implausible values ({len(impl)} pollutants affected):")
            for f in impl:
                print(f"    {f['pollutant']:<8} {f['n_flagged']} rows > {f['threshold']} "
                      f"(max={f['max_seen']:.0f})")
                issues.append(f"Implausible {f['pollutant']}: {f['n_flagged']} rows")
        else:
            print("\n  Implausible values: none")

        if outl:
            print(f"\n  Outliers (>{OUTLIER_SIGMA}σ) ({len(outl)} pollutants):")
            for f in outl:
                print(f"    {f['pollutant']:<8} {f['n_outliers']} rows > {f['cutoff']:.0f} "
                      f"(mean={f['mean']:.0f}, std={f['std']:.0f}, max={f['max_val']:.0f})")
                issues.append(f"Outliers in {f['pollutant']}: {f['n_outliers']}")
        else:
            print("\n  Outliers: none")

        if stuck:
            print(f"\n  Stuck sensors: {len(stuck)} cases")
            for s in stuck[:5]:
                print(f"    {s['station'][:30]:<30} {s['pollutant']:<8} "
                      f"value={s['stuck_value']:.1f} x{s['repeat_count']}")
            issues.append(f"Stuck sensors: {len(stuck)} cases")
        else:
            print("\n  Stuck sensors: none")

        print(f"\n  Station coverage:")
        print(f"    Total stations         : {cov['total_stations']}")
        print(f"    Avg pollutants/station : {cov['avg_pollutants_per_station']:.1f} / {len(POLLUTANTS)}")
        print(f"    Sparse stations (<50%) : {cov['sparse_stations']}")
        if cov['sparse_stations'] > 0:
            issues.append(f"Sparse stations: {cov['sparse_stations']}")

        print()
        if issues:
            print(f"  SUMMARY: {len(issues)} issue(s) found — run with --fix to clean")
        else:
            print("  SUMMARY: No issues found. Data looks clean.")
        print()

    _plot_quality(df, outl)

    return {
        'total_rows': len(df),
        'issues': issues,
        'missing': missing,
        'negatives': neg,
        'implausible': impl,
        'outliers': outl,
        'stuck_sensors': stuck,
        'coverage': cov,
    }


def _plot_quality(df, outliers_info):
    """Visualise per-pollutant distributions with outlier cutoffs."""
    available = [p for p in POLLUTANTS if (df['pollutant_id'] == p).any()]
    n = len(available)
    if n == 0:
        return

    cols_l, rows_l = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows_l, cols_l, figsize=(14, rows_l * 3))
    fig.suptitle('Data Quality — Pollutant Distributions', fontsize=14, fontweight='bold')
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    ax_list = axes.flat if rows_l > 1 else (axes if n > 1 else [axes])

    outlier_map = {o['pollutant']: o['cutoff'] for o in (outliers_info or [])}

    for ax, p, color in zip(ax_list, available, palette):
        data = df[df['pollutant_id'] == p]['pollutant_avg'].dropna()
        ax.hist(data, bins=30, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(data.median(), color='black', lw=1.5, ls='--',
                   label=f'Median={data.median():.0f}')
        if p in outlier_map:
            ax.axvline(outlier_map[p], color='red', lw=1.5, ls=':',
                       label=f'Outlier cutoff={outlier_map[p]:.0f}')
        ax.set_title(p); ax.set_xlabel('µg/m³'); ax.set_ylabel('Count')
        ax.legend(fontsize=7)

    for ax in list(ax_list)[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'dq_pollutant_check.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/dq_pollutant_check.png")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AQI Data Quality Validator')
    parser.add_argument('--file', default=RT_DATA_PATH, help='CSV file to validate')
    parser.add_argument('--fix',  action='store_true',  help='Write cleaned CSV alongside original')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
        sys.exit(1)

    df = pd.read_csv(args.file)
    report = run_report(df, verbose=True)

    if args.fix and report['issues']:
        clean_df, removed = clean_data(df)
        out_path = args.file.replace('.csv', '_cleaned.csv')
        clean_df.to_csv(out_path, index=False)
        print(f"  Cleaned data: {removed} rows removed -> {out_path}")
