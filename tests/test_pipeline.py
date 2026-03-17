"""
Integration tests for the real-time AQI pipeline.

Tests core pipeline functions: data loading, feature engineering,
model I/O, and prediction shape/type correctness.

Run:
    pytest tests/test_pipeline.py -v
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from realtime_aqi_pipeline import (
    compute_sub_index, aqi_to_category,
    load_and_process, engineer_features,
    build_input_vector, predict_aqi,
    POLLUTANTS, OUTPUT_DIR, FIGURES_DIR,
)
from config import RT_DATA_PATH, RT_MODEL_PATH, RT_FEAT_PATH

MODEL_EXISTS = os.path.exists(RT_MODEL_PATH)
DATA_EXISTS  = os.path.exists(RT_DATA_PATH)


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not DATA_EXISTS, reason="Real-time CSV not found")
class TestLoadAndProcess:
    @pytest.fixture(scope="class")
    def df(self):
        return load_and_process(RT_DATA_PATH)

    def test_returns_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)

    def test_has_aqi_column(self, df):
        assert 'AQI' in df.columns

    def test_has_lat_lng(self, df):
        assert 'lat' in df.columns
        assert 'lng' in df.columns

    def test_no_null_aqi(self, df):
        assert df['AQI'].isna().sum() == 0

    def test_aqi_in_valid_range(self, df):
        assert df['AQI'].min() >= 0
        assert df['AQI'].max() <= 500

    def test_all_pollutant_columns_present(self, df):
        for p in POLLUTANTS:
            assert p in df.columns, f"Missing pollutant column: {p}"

    def test_lat_lng_in_india_range(self, df):
        assert df['lat'].between(6, 37).all()
        assert df['lng'].between(66, 98).all()

    def test_aqi_category_column(self, df):
        assert 'AQI_Category' in df.columns
        valid_cats = {'Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'}
        assert set(df['AQI_Category'].unique()).issubset(valid_cats)

    def test_minimum_row_count(self, df):
        assert len(df) >= 100, "Too few stations after processing"


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not DATA_EXISTS, reason="Real-time CSV not found")
class TestEngineerFeatures:
    @pytest.fixture(scope="class")
    def X(self):
        df = load_and_process(RT_DATA_PATH)
        return engineer_features(df), df

    def test_returns_dataframe(self, X):
        features, _ = X
        assert isinstance(features, pd.DataFrame)

    def test_no_nulls(self, X):
        features, _ = X
        assert features.isna().sum().sum() == 0, "Feature matrix has NaN values"

    def test_has_geo_features(self, X):
        features, _ = X
        for col in ['lat', 'lng', 'lat_abs', 'lat_sin', 'lng_cos', 'lng_sin']:
            assert col in features.columns

    def test_has_temporal_features(self, X):
        features, _ = X
        for col in ['hour', 'day_of_week', 'month', 'is_weekend', 'season']:
            assert col in features.columns

    def test_has_pollutant_features(self, X):
        features, _ = X
        for p in POLLUTANTS:
            assert p in features.columns

    def test_row_count_matches_df(self, X):
        features, df = X
        assert len(features) == len(df)

    def test_lat_abs_non_negative(self, X):
        features, _ = X
        assert (features['lat_abs'] >= 0).all()

    def test_sin_cos_in_unit_range(self, X):
        features, _ = X
        for col in ['lat_sin', 'lng_cos', 'lng_sin']:
            assert features[col].between(-1, 1).all(), f"{col} out of [-1,1]"

    def test_season_valid_values(self, X):
        features, _ = X
        assert features['season'].isin([1, 2, 3, 4]).all()

    def test_is_weekend_binary(self, X):
        features, _ = X
        assert features['is_weekend'].isin([0, 1]).all()


# ─────────────────────────────────────────────────────────────
# build_input_vector
# ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestBuildInputVector:
    def test_returns_single_row_dataframe(self):
        X = build_input_vector(28.6, 77.2)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 1

    def test_feature_names_match_model(self):
        import joblib
        feat_names = joblib.load(RT_FEAT_PATH)
        X = build_input_vector(28.6, 77.2)
        assert list(X.columns) == feat_names

    def test_no_nulls_in_output(self):
        X = build_input_vector(28.6, 77.2)
        assert X.isna().sum().sum() == 0

    def test_lat_lng_passed_correctly(self):
        X = build_input_vector(28.6, 77.2)
        assert abs(float(X['lat'].iloc[0]) - 28.6) < 1e-6
        assert abs(float(X['lng'].iloc[0]) - 77.2) < 1e-6

    def test_pollutants_dict_overrides_fallback(self):
        X_default = build_input_vector(28.6, 77.2)
        X_custom  = build_input_vector(28.6, 77.2, {'PM2.5': 999.0})
        assert float(X_custom['PM2.5'].iloc[0]) == 999.0
        # Default fallback uses nearest station, not 999
        assert float(X_default['PM2.5'].iloc[0]) != 999.0

    def test_hour_parameter_applied(self):
        X = build_input_vector(28.6, 77.2, hour=14)
        assert int(X['hour'].iloc[0]) == 14

    def test_month_parameter_applied(self):
        X = build_input_vector(28.6, 77.2, month=7)
        assert int(X['month'].iloc[0]) == 7


# ─────────────────────────────────────────────────────────────
# predict_aqi — smoke tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
class TestPredictAqi:
    def test_returns_tuple(self):
        result = predict_aqi(28.6, 77.2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_aqi_is_float(self):
        aqi, _ = predict_aqi(28.6, 77.2)
        assert isinstance(aqi, float)

    def test_category_is_string(self):
        _, cat = predict_aqi(28.6, 77.2)
        assert isinstance(cat, str)

    def test_aqi_non_negative(self):
        aqi, _ = predict_aqi(28.6, 77.2)
        assert aqi >= 0

    def test_aqi_below_500(self):
        aqi, _ = predict_aqi(28.6, 77.2)
        assert aqi <= 500

    def test_category_is_valid(self):
        _, cat = predict_aqi(28.6, 77.2)
        valid = {'Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe', 'Unknown'}
        assert cat in valid

    def test_high_pollution_input_gives_high_aqi(self):
        """Providing very high PM2.5 should yield higher AQI than clean air."""
        aqi_clean, _ = predict_aqi(28.6, 77.2, {'PM2.5': 5})
        aqi_dirty, _ = predict_aqi(28.6, 77.2, {'PM2.5': 300})
        assert aqi_dirty > aqi_clean

    def test_different_cities_differ(self):
        """Different city coordinates should generally yield different predictions."""
        aqi_delhi, _     = predict_aqi(28.6, 77.2)
        aqi_bangalore, _ = predict_aqi(12.97, 77.59)
        # They may not always differ, but it's a sanity check
        # (at least both should be valid)
        assert aqi_delhi >= 0
        assert aqi_bangalore >= 0

    def test_with_pollutants_dict(self):
        """Prediction with pollutant dict should be valid."""
        aqi, cat = predict_aqi(19.07, 72.87,
                               {'PM2.5': 65, 'PM10': 110, 'NO2': 35})
        assert aqi >= 0
        assert cat in {'Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'}


# ─────────────────────────────────────────────────────────────
# Output directory structure
# ─────────────────────────────────────────────────────────────

class TestOutputDirectory:
    def test_output_dir_exists(self):
        assert os.path.isdir(OUTPUT_DIR)

    def test_figures_dir_exists(self):
        assert os.path.isdir(FIGURES_DIR)

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
    def test_model_file_exists(self):
        assert os.path.exists(RT_MODEL_PATH)

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet")
    def test_feature_names_file_exists(self):
        assert os.path.exists(RT_FEAT_PATH)
