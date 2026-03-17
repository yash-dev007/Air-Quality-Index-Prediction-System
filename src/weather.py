"""
Weather Feature Fetcher
========================
Fetches current weather (temp, humidity, wind speed, pressure) from
OpenWeatherMap free API for a given lat/lng.

API key setup (choose one):
  1. Set env var:  OWM_API_KEY=your_key
  2. Streamlit:    st.secrets["OWM_API_KEY"]
  3. Pass directly: fetch_weather(lat, lng, api_key="your_key")

Free tier: 60 calls/min, no credit card needed.
Sign up at https://openweathermap.org/api
"""

import os
import requests

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

# Weather feature names (must match engineer_features order if used in training)
WEATHER_FEATURES = ['temp_c', 'humidity', 'wind_speed', 'pressure']


def fetch_weather(lat, lng, api_key=None):
    """
    Fetch current weather for a location.
    Returns dict with temp_c, humidity, wind_speed, pressure — or None on failure.
    """
    key = api_key or os.environ.get('OWM_API_KEY')
    if not key:
        return None

    try:
        resp = requests.get(
            OWM_URL,
            params={'lat': lat, 'lon': lng, 'appid': key, 'units': 'metric'},
            timeout=8,
        )
        resp.raise_for_status()
        d = resp.json()
        return {
            'temp_c'    : d['main']['temp'],
            'humidity'  : d['main']['humidity'],
            'wind_speed': d['wind']['speed'],
            'pressure'  : d['main']['pressure'],
            'description': d['weather'][0]['description'].title(),
            'city_name' : d.get('name', ''),
        }
    except (requests.RequestException, KeyError):
        return None


def weather_impact_note(weather):
    """
    Return a short human-readable note about how current weather
    may be affecting AQI.
    """
    if not weather:
        return ""
    notes = []
    if weather['wind_speed'] < 2:
        notes.append("Low wind — pollutants may accumulate.")
    elif weather['wind_speed'] > 8:
        notes.append("High wind — pollutants likely dispersed.")
    if weather['humidity'] > 80:
        notes.append("High humidity — PM2.5 can spike due to hygroscopic growth.")
    if weather['temp_c'] > 35:
        notes.append("High temperature — ozone formation likely elevated.")
    if weather['temp_c'] < 5:
        notes.append("Cold weather — temperature inversions may trap pollution.")
    return "  ".join(notes) if notes else "Weather conditions appear neutral for AQI."
