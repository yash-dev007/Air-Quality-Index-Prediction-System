"""
API endpoint tests using FastAPI TestClient.

Run:
    pytest tests/test_api.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Only run API tests if model exists
from config import RT_MODEL_PATH
MODEL_EXISTS = os.path.exists(RT_MODEL_PATH)

pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS,
    reason="Model not trained yet — run: python src/realtime_aqi_pipeline.py"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)


class TestHealth:
    def test_root_returns_ok(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data


class TestPredictGET:
    def test_valid_delhi(self, client):
        resp = client.get("/predict?lat=28.6139&lng=77.2090")
        assert resp.status_code == 200
        data = resp.json()
        assert "aqi" in data
        assert "category" in data
        assert data["aqi"] >= 0

    def test_valid_mumbai(self, client):
        resp = client.get("/predict?lat=19.076&lng=72.8777")
        assert resp.status_code == 200
        assert resp.json()["aqi"] >= 0

    def test_lat_out_of_range(self, client):
        resp = client.get("/predict?lat=50.0&lng=77.2")
        assert resp.status_code == 422  # Validation error

    def test_lng_out_of_range(self, client):
        resp = client.get("/predict?lat=28.6&lng=10.0")
        assert resp.status_code == 422

    def test_missing_lat(self, client):
        resp = client.get("/predict?lng=77.2")
        assert resp.status_code == 422

    def test_with_hour(self, client):
        resp = client.get("/predict?lat=28.6&lng=77.2&hour=14")
        assert resp.status_code == 200

    def test_with_month(self, client):
        resp = client.get("/predict?lat=28.6&lng=77.2&month=12")
        assert resp.status_code == 200

    def test_response_has_color(self, client):
        resp = client.get("/predict?lat=28.6&lng=77.2")
        data = resp.json()
        assert "color" in data
        assert data["color"].startswith("#")

    def test_response_category_valid(self, client):
        resp = client.get("/predict?lat=28.6&lng=77.2")
        cat  = resp.json()["category"]
        valid = {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}
        assert cat in valid


class TestPredictPOST:
    def test_basic_post(self, client):
        resp = client.post("/predict", json={"lat": 28.6, "lng": 77.2})
        assert resp.status_code == 200
        assert resp.json()["aqi"] >= 0

    def test_with_pollutants(self, client):
        resp = client.post("/predict", json={
            "lat": 28.6, "lng": 77.2,
            "PM2.5": 80.0, "PM10": 150.0, "NO2": 45.0
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["aqi"] >= 0
        assert data["sub_indices"] is not None

    def test_high_pollution_yields_high_aqi(self, client):
        clean = client.post("/predict", json={"lat": 28.6, "lng": 77.2,
                                               "PM2.5": 5.0}).json()
        dirty = client.post("/predict", json={"lat": 28.6, "lng": 77.2,
                                               "PM2.5": 300.0}).json()
        assert dirty["aqi"] > clean["aqi"]


class TestPredictInterval:
    def test_returns_interval_fields(self, client):
        resp = client.get("/predict/interval?lat=28.6&lng=77.2")
        assert resp.status_code == 200
        data = resp.json()
        assert "aqi" in data
        # lower_80/upper_80 may be None if quantile models not trained
        assert "lower_80" in data
        assert "upper_80" in data


class TestCities:
    def test_list_cities(self, client):
        resp = client.get("/cities")
        assert resp.status_code == 200
        data = resp.json()
        assert "cities" in data
        assert len(data["cities"]) > 0

    def test_city_has_required_fields(self, client):
        resp = client.get("/cities")
        city = resp.json()["cities"][0]
        for field in ["city", "lat", "lng", "aqi", "category", "color"]:
            assert field in city

    def test_get_delhi(self, client):
        resp = client.get("/cities/Delhi")
        assert resp.status_code == 200
        data = resp.json()
        assert data["city"] == "Delhi"
        assert data["aqi"] >= 0

    def test_get_unknown_city_404(self, client):
        resp = client.get("/cities/FakeCity123")
        assert resp.status_code == 404

    def test_cities_sorted_by_aqi_desc(self, client):
        cities = client.get("/cities").json()["cities"]
        aqis = [c["aqi"] for c in cities]
        assert aqis == sorted(aqis, reverse=True)


class TestNAQICompute:
    def test_compute_from_pm25_only(self, client):
        resp = client.get("/naqi/compute?PM2.5=65")
        assert resp.status_code == 200
        data = resp.json()
        assert data["aqi"] > 0
        assert "dominant_pollutant" in data
        assert data["dominant_pollutant"] == "PM2.5"

    def test_no_pollutants_returns_400(self, client):
        resp = client.get("/naqi/compute")
        assert resp.status_code == 400

    def test_who_comparison_present(self, client):
        resp = client.get("/naqi/compute?PM2.5=65&NO2=45")
        data = resp.json()
        assert "who_comparison" in data
        assert len(data["who_comparison"]) == 2

    def test_high_pm25_exceeds_who(self, client):
        resp  = client.get("/naqi/compute?PM2.5=100")
        comps = resp.json()["who_comparison"]
        pm25_entry = next(c for c in comps if c["pollutant"] == "PM2.5")
        assert pm25_entry["exceeds"] is True
        assert pm25_entry["times_over_guideline"] == 20.0
