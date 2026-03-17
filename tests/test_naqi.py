"""
Unit tests for India NAQI computation.

These tests are the most critical in the project — a wrong breakpoint
boundary silently corrupts every prediction, alert, and forecast.

Run:
    pytest tests/test_naqi.py -v
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from realtime_aqi_pipeline import compute_sub_index, aqi_to_category, aqi_to_color


# ─────────────────────────────────────────────────────────────
# compute_sub_index — boundary and mid-range values
# ─────────────────────────────────────────────────────────────

class TestPM25:
    """PM2.5 NAQI sub-index (c_breaks: 0,30,60,90,120,250,500)"""

    def test_zero_is_zero(self):
        assert compute_sub_index(0, 'PM2.5') == 0

    def test_lower_boundary_good(self):
        # At exactly 30 µg/m³ -> sub-index = 50
        assert compute_sub_index(30, 'PM2.5') == 50.0

    def test_midpoint_satisfactory(self):
        # 45 µg/m³: between 30–60 -> (100-50)/(60-30) * (45-30) + 50 = 75
        assert compute_sub_index(45, 'PM2.5') == 75.0

    def test_lower_boundary_satisfactory(self):
        assert compute_sub_index(60, 'PM2.5') == 100.0

    def test_moderate_midpoint(self):
        # 75 µg/m³: between 60–90 -> (200-100)/(90-60) * (75-60) + 100 = 150
        assert compute_sub_index(75, 'PM2.5') == 150.0

    def test_poor_boundary(self):
        assert compute_sub_index(120, 'PM2.5') == 300.0

    def test_very_poor_midpoint(self):
        # 185 µg/m³: between 120–250 -> (400-300)/(250-120) * (185-120) + 300 = 350
        result = compute_sub_index(185, 'PM2.5')
        assert abs(result - 350.0) < 0.5

    def test_severe_boundary(self):
        assert compute_sub_index(250, 'PM2.5') == 400.0

    def test_beyond_max_clamps_to_500(self):
        assert compute_sub_index(999, 'PM2.5') == 500

    def test_negative_returns_nan(self):
        assert math.isnan(compute_sub_index(-1, 'PM2.5'))

    def test_nan_returns_nan(self):
        import numpy as np
        assert math.isnan(compute_sub_index(np.nan, 'PM2.5'))


class TestPM10:
    """PM10 NAQI sub-index (c_breaks: 0,50,100,250,350,430,600)"""

    def test_zero(self):
        assert compute_sub_index(0, 'PM10') == 0

    def test_good_upper(self):
        assert compute_sub_index(50, 'PM10') == 50.0

    def test_satisfactory_upper(self):
        assert compute_sub_index(100, 'PM10') == 100.0

    def test_moderate_upper(self):
        assert compute_sub_index(250, 'PM10') == 200.0

    def test_poor_upper(self):
        assert compute_sub_index(350, 'PM10') == 300.0

    def test_very_poor_upper(self):
        assert compute_sub_index(430, 'PM10') == 400.0

    def test_beyond_max(self):
        assert compute_sub_index(700, 'PM10') == 500


class TestNO2:
    """NO2 NAQI sub-index (c_breaks: 0,40,80,180,280,400,600)"""

    def test_zero(self):
        assert compute_sub_index(0, 'NO2') == 0

    def test_good_upper(self):
        assert compute_sub_index(40, 'NO2') == 50.0

    def test_satisfactory_upper(self):
        assert compute_sub_index(80, 'NO2') == 100.0

    def test_moderate_upper(self):
        assert compute_sub_index(180, 'NO2') == 200.0

    def test_beyond_max(self):
        assert compute_sub_index(700, 'NO2') == 500


class TestSO2:
    """SO2 NAQI sub-index (c_breaks: 0,40,80,380,800,1600,2100)"""

    def test_good_upper(self):
        assert compute_sub_index(40, 'SO2') == 50.0

    def test_moderate_range(self):
        # Between 80 and 380 -> large range
        result = compute_sub_index(230, 'SO2')
        # (200-100)/(380-80) * (230-80) + 100 = 150
        assert abs(result - 150.0) < 0.5

    def test_beyond_max(self):
        assert compute_sub_index(3000, 'SO2') == 500


class TestOZONE:
    """OZONE NAQI sub-index (c_breaks: 0,50,100,168,208,748,1000)"""

    def test_good_upper(self):
        assert compute_sub_index(50, 'OZONE') == 50.0

    def test_satisfactory_upper(self):
        assert compute_sub_index(100, 'OZONE') == 100.0

    def test_beyond_max(self):
        assert compute_sub_index(1100, 'OZONE') == 500


class TestNH3:
    """NH3 NAQI sub-index (c_breaks: 0,200,400,800,1200,1800,2400)"""

    def test_low_value_good(self):
        # 5 µg/m³ is well within Good range
        result = compute_sub_index(5, 'NH3')
        assert result < 50

    def test_good_upper(self):
        assert compute_sub_index(200, 'NH3') == 50.0

    def test_satisfactory_upper(self):
        assert compute_sub_index(400, 'NH3') == 100.0

    def test_beyond_max(self):
        assert compute_sub_index(3000, 'NH3') == 500


class TestCO:
    """CO NAQI sub-index (c_breaks: 0,1000,2000,10000,17000,34000,50000 µg/m³)"""

    def test_low_value_good(self):
        result = compute_sub_index(100, 'CO')
        assert result < 50

    def test_good_upper(self):
        assert compute_sub_index(1000, 'CO') == 50.0

    def test_satisfactory_upper(self):
        assert compute_sub_index(2000, 'CO') == 100.0

    def test_beyond_max(self):
        assert compute_sub_index(60000, 'CO') == 500


class TestUnknownPollutant:
    def test_unknown_returns_nan(self):
        assert math.isnan(compute_sub_index(100, 'UNKNOWN'))


# ─────────────────────────────────────────────────────────────
# aqi_to_category — boundary values
# ─────────────────────────────────────────────────────────────

class TestAqiToCategory:
    def test_zero_is_good(self):
        assert aqi_to_category(0) == 'Good'

    def test_50_is_good(self):
        assert aqi_to_category(50) == 'Good'

    def test_51_is_satisfactory(self):
        assert aqi_to_category(51) == 'Satisfactory'

    def test_100_is_satisfactory(self):
        assert aqi_to_category(100) == 'Satisfactory'

    def test_101_is_moderate(self):
        assert aqi_to_category(101) == 'Moderate'

    def test_200_is_moderate(self):
        assert aqi_to_category(200) == 'Moderate'

    def test_201_is_poor(self):
        assert aqi_to_category(201) == 'Poor'

    def test_300_is_poor(self):
        assert aqi_to_category(300) == 'Poor'

    def test_301_is_very_poor(self):
        assert aqi_to_category(301) == 'Very Poor'

    def test_400_is_very_poor(self):
        assert aqi_to_category(400) == 'Very Poor'

    def test_401_is_severe(self):
        assert aqi_to_category(401) == 'Severe'

    def test_500_is_severe(self):
        assert aqi_to_category(500) == 'Severe'

    def test_above_500_is_severe(self):
        assert aqi_to_category(999) == 'Severe'

    def test_nan_is_unknown(self):
        import numpy as np
        assert aqi_to_category(np.nan) == 'Unknown'


# ─────────────────────────────────────────────────────────────
# aqi_to_color — sanity checks
# ─────────────────────────────────────────────────────────────

class TestAqiToColor:
    def test_good_is_green(self):
        assert aqi_to_color(25) == '#2ecc71'

    def test_severe_is_purple(self):
        assert aqi_to_color(450) == '#8e44ad'

    def test_returns_hex_string(self):
        color = aqi_to_color(150)
        assert color.startswith('#')
        assert len(color) == 7


# ─────────────────────────────────────────────────────────────
# Overall AQI = max of sub-indices (India NAQI rule)
# ─────────────────────────────────────────────────────────────

class TestNAQIRule:
    def test_aqi_equals_max_sub_index(self):
        """Verify the NAQI rule: overall AQI = max of sub-indices."""
        pollutants = {'PM2.5': 45, 'PM10': 80, 'NO2': 20}
        sub_indices = {p: compute_sub_index(v, p) for p, v in pollutants.items()}
        overall = max(sub_indices.values())
        # PM2.5=45 -> 75, PM10=80 -> 80, NO2=20 -> ~25
        assert overall == sub_indices['PM10']  # PM10 dominates

    def test_category_matches_max_sub_index(self):
        """Category of NAQI should match category of max sub-index."""
        pm25_sub = compute_sub_index(85, 'PM2.5')  # -> ~158.3 (Moderate)
        assert aqi_to_category(pm25_sub) == 'Moderate'
