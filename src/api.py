"""
FastAPI REST Endpoint for AQI Predictions
==========================================
Exposes the trained model as a JSON REST API.

Start server:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Or via Makefile:
    make api

Endpoints:
    GET  /                         health check
    GET  /predict                  predict AQI for lat/lng
    POST /predict                  predict AQI with full input body
    GET  /predict/interval         predict with 80% confidence interval
    GET  /cities                   list all supported cities with predictions
    GET  /cities/{city_name}       get detailed prediction for a named city
    GET  /health                   service health + model info
    GET  /docs                     auto-generated Swagger UI
"""

import os
import sys
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from realtime_aqi_pipeline import (
    predict_aqi, aqi_to_category, aqi_to_color,
    compute_sub_index, POLLUTANTS, OUTPUT_DIR,
)
from config import INDIA_CITIES, WHO_GUIDELINES, RT_MODEL_PATH
import joblib
import numpy as np

app = FastAPI(
    title="India AQI Prediction API",
    description=(
        "Predicts India National Air Quality Index (NAQI) from geographic coordinates, "
        "temporal features, and optional pollutant concentrations. "
        "Trained on real-time CPCB monitoring station data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────
class PredictRequest(BaseModel):
    lat:   float = Field(..., ge=6.0, le=37.0, description="Latitude (India: 6–37)")
    lng:   float = Field(..., ge=66.0, le=98.0, description="Longitude (India: 66–98)")
    hour:  Optional[int]   = Field(None, ge=0, le=23,  description="Hour of day (0–23)")
    month: Optional[int]   = Field(None, ge=1, le=12,  description="Month (1–12)")
    PM2_5: Optional[float] = Field(None, ge=0, alias="PM2.5", description="PM2.5 µg/m³")
    PM10:  Optional[float] = Field(None, ge=0, description="PM10 µg/m³")
    NO2:   Optional[float] = Field(None, ge=0, description="NO2 µg/m³")
    SO2:   Optional[float] = Field(None, ge=0, description="SO2 µg/m³")
    OZONE: Optional[float] = Field(None, ge=0, description="OZONE µg/m³")
    NH3:   Optional[float] = Field(None, ge=0, description="NH3 µg/m³")
    CO:    Optional[float] = Field(None, ge=0, description="CO µg/m³")

    class Config:
        populate_by_name = True


class PredictResponse(BaseModel):
    aqi:        float
    category:   str
    color:      str
    lat:        float
    lng:        float
    sub_indices: Optional[Dict[str, float]] = None


class IntervalResponse(PredictResponse):
    lower_80:  Optional[float] = None
    upper_80:  Optional[float] = None
    interval:  Optional[str]   = None


# ── Helper ────────────────────────────────────────────────────
def _build_pollutants_dict(req) -> Optional[dict]:
    p = {}
    fields = {
        'PM2.5': getattr(req, 'PM2_5', None) or getattr(req, 'PM2.5', None),
        'PM10' : req.PM10,
        'NO2'  : req.NO2,
        'SO2'  : req.SO2,
        'OZONE': req.OZONE,
        'NH3'  : req.NH3,
        'CO'   : req.CO,
    }
    for k, v in fields.items():
        if v is not None:
            p[k] = v
    return p if p else None


def _sub_indices(p_dict: dict) -> dict:
    return {
        p: round(float(compute_sub_index(v, p)), 1)
        for p, v in p_dict.items()
        if not np.isnan(compute_sub_index(v, p))
    }


def _who_comparison(p_dict: dict) -> list:
    result = []
    for p, val in p_dict.items():
        guideline = WHO_GUIDELINES.get(p)
        if guideline:
            times_over = round(val / guideline, 1)
            result.append({
                'pollutant': p,
                'measured_ugm3': val,
                'who_guideline_ugm3': guideline,
                'times_over_guideline': times_over,
                'exceeds': val > guideline,
            })
    return result


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "India AQI Prediction API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    model_ok = os.path.exists(RT_MODEL_PATH)
    return {
        "status": "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_path": RT_MODEL_PATH,
    }


@app.get("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_get(
    lat:   float = Query(..., ge=6.0,  le=37.0),
    lng:   float = Query(..., ge=66.0, le=98.0),
    hour:  Optional[int]   = Query(None, ge=0, le=23),
    month: Optional[int]   = Query(None, ge=1, le=12),
):
    """Quick prediction from lat/lng (uses nearest training station pollutants)."""
    try:
        aqi, cat = predict_aqi(lat, lng, hour=hour, month=month)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return PredictResponse(
        aqi=aqi, category=cat, color=aqi_to_color(aqi), lat=lat, lng=lng
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_post(req: PredictRequest):
    """Full prediction with optional pollutant concentrations."""
    try:
        p_dict = _build_pollutants_dict(req)
        aqi, cat = predict_aqi(req.lat, req.lng, p_dict, hour=req.hour, month=req.month)
        sub = _sub_indices(p_dict) if p_dict else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return PredictResponse(
        aqi=aqi, category=cat, color=aqi_to_color(aqi),
        lat=req.lat, lng=req.lng, sub_indices=sub,
    )


@app.get("/predict/interval", response_model=IntervalResponse, tags=["Prediction"])
def predict_interval(
    lat:   float = Query(..., ge=6.0,  le=37.0),
    lng:   float = Query(..., ge=66.0, le=98.0),
    hour:  Optional[int]   = Query(None),
    month: Optional[int]   = Query(None),
):
    """Prediction with 80% confidence interval (quantile regression)."""
    try:
        aqi, cat = predict_aqi(lat, lng, hour=hour, month=month)
        lower = upper = None
        from realtime_aqi_pipeline import build_input_vector
        from config import RT_LOWER_MODEL_PATH, RT_UPPER_MODEL_PATH
        X = build_input_vector(lat, lng, hour=hour, month=month)
        if os.path.exists(RT_LOWER_MODEL_PATH) and os.path.exists(RT_UPPER_MODEL_PATH):
            lower = max(0.0, round(float(joblib.load(RT_LOWER_MODEL_PATH).predict(X)[0]), 1))
            upper = max(0.0, round(float(joblib.load(RT_UPPER_MODEL_PATH).predict(X)[0]), 1))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IntervalResponse(
        aqi=aqi, category=cat, color=aqi_to_color(aqi), lat=lat, lng=lng,
        lower_80=lower, upper_80=upper,
        interval=f"{lower:.0f}–{upper:.0f}" if lower is not None else None,
    )


@app.get("/cities", tags=["Cities"])
def list_cities():
    """Return all supported cities with their current predicted AQI."""
    results = []
    for name, lat, lng in INDIA_CITIES:
        aqi, cat = predict_aqi(lat, lng)
        results.append({
            "city": name, "lat": lat, "lng": lng,
            "aqi": aqi, "category": cat, "color": aqi_to_color(aqi),
        })
    results.sort(key=lambda x: x["aqi"], reverse=True)
    return {"cities": results, "total": len(results)}


@app.get("/cities/{city_name}", tags=["Cities"])
def get_city(city_name: str):
    """Get detailed prediction for a named city including pollutant sub-indices."""
    match = next(((n, la, lo) for n, la, lo in INDIA_CITIES
                   if n.lower() == city_name.lower()), None)
    if match is None:
        available = [n for n, _, _ in INDIA_CITIES]
        raise HTTPException(status_code=404,
                            detail=f"City '{city_name}' not found. Available: {available}")
    name, lat, lng = match
    aqi, cat = predict_aqi(lat, lng)

    # Get nearest station data for sub-indices
    from realtime_aqi_pipeline import build_input_vector
    import pandas as pd
    from config import RT_STATIONS_PATH
    sub_indices = {}
    if os.path.exists(RT_STATIONS_PATH):
        stations = pd.read_csv(RT_STATIONS_PATH)
        dists = np.sqrt((stations['lat'] - lat)**2 + (stations['lng'] - lng)**2)
        nearest = stations.iloc[dists.idxmin()]
        for p in POLLUTANTS:
            if p in nearest and not pd.isna(nearest[p]):
                si = compute_sub_index(float(nearest[p]), p)
                if not np.isnan(si):
                    sub_indices[p] = {'concentration_ugm3': float(nearest[p]),
                                      'sub_index': round(float(si), 1)}

    return {
        "city": name, "lat": lat, "lng": lng,
        "aqi": aqi, "category": cat, "color": aqi_to_color(aqi),
        "sub_indices": sub_indices,
        "who_comparison": _who_comparison({p: v['concentration_ugm3']
                                            for p, v in sub_indices.items()}),
    }


@app.get("/naqi/compute", tags=["Utilities"])
def compute_naqi(
    PM2_5: Optional[float] = Query(None, alias="PM2.5"),
    PM10:  Optional[float] = Query(None),
    NO2:   Optional[float] = Query(None),
    SO2:   Optional[float] = Query(None),
    OZONE: Optional[float] = Query(None),
    NH3:   Optional[float] = Query(None),
    CO:    Optional[float] = Query(None),
):
    """Compute India NAQI directly from pollutant concentrations (no ML model needed)."""
    inputs = {'PM2.5': PM2_5, 'PM10': PM10, 'NO2': NO2,
              'SO2': SO2, 'OZONE': OZONE, 'NH3': NH3, 'CO': CO}
    sub_indices = {}
    for p, val in inputs.items():
        if val is not None:
            si = compute_sub_index(val, p)
            if not np.isnan(si):
                sub_indices[p] = round(float(si), 1)

    if not sub_indices:
        raise HTTPException(status_code=400, detail="Provide at least one pollutant concentration.")

    overall_aqi = max(sub_indices.values())
    dominant    = max(sub_indices, key=sub_indices.get)
    return {
        "aqi": round(overall_aqi, 1),
        "category": aqi_to_category(overall_aqi),
        "color": aqi_to_color(overall_aqi),
        "dominant_pollutant": dominant,
        "sub_indices": sub_indices,
        "who_comparison": _who_comparison(
            {p: v for p, v in inputs.items() if v is not None}
        ),
    }


# ── Run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
