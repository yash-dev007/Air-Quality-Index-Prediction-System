"""
Historical AQI Data Collector
==============================
Fetches air quality measurements from the free OpenAQ v3 API for Indian cities
and appends them to a growing CSV.  Run daily (e.g., via Task Scheduler or cron)
to build a time-series dataset for temporal modelling.

Usage:
    python src/fetch_historical.py                  # fetch for default cities
    python src/fetch_historical.py --cities Delhi Mumbai Chennai
    python src/fetch_historical.py --lat 28.6 --lng 77.2   # single point
"""

import os
import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timezone

_ROOT    = os.path.join(os.path.dirname(__file__), '..')
OUT_FILE = os.path.join(_ROOT, 'data', 'historical_aqi.csv')

# Major Indian cities: (name, lat, lng)
DEFAULT_CITIES = [
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
    ("Visakhapatnam",  17.6868, 83.2185),
    ("Coimbatore",     11.0168, 76.9558),
    ("Surat",          21.1702, 72.8311),
    ("Vadodara",       22.3072, 73.1812),
]

PARAM_MAP = {
    'pm25': 'PM2.5', 'pm2.5': 'PM2.5',
    'pm10': 'PM10',
    'no2' : 'NO2',
    'so2' : 'SO2',
    'o3'  : 'OZONE',
    'nh3' : 'NH3',
    'co'  : 'CO',
}


def fetch_station(lat, lng, radius_km=30):
    """Fetch latest pollutant readings near a location. Returns a dict or None."""
    try:
        # Find nearest station
        loc_resp = requests.get(
            "https://api.openaq.org/v3/locations",
            params={
                'coordinates': f'{lat},{lng}',
                'radius'     : radius_km * 1000,
                'limit'      : 1,
                'order_by'   : 'distance',
            },
            headers={'Accept': 'application/json'},
            timeout=15,
        )
        loc_resp.raise_for_status()
        locations = loc_resp.json().get('results', [])
        if not locations:
            return None

        loc = locations[0]
        loc_id   = loc['id']
        loc_name = loc.get('name', 'Unknown')

        # Fetch latest measurements
        meas_resp = requests.get(
            f"https://api.openaq.org/v3/locations/{loc_id}/latest",
            headers={'Accept': 'application/json'},
            timeout=15,
        )
        meas_resp.raise_for_status()
        measurements = meas_resp.json().get('results', [])

        row = {
            'timestamp'   : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'station_name': loc_name,
            'lat'         : lat,
            'lng'         : lng,
        }
        for m in measurements:
            param = m.get('parameter', '').lower()
            value = m.get('value')
            if param in PARAM_MAP and value is not None and value >= 0:
                row[PARAM_MAP[param]] = value

        return row

    except (requests.RequestException, KeyError, ValueError):
        return None


def fetch_all(cities):
    rows = []
    for name, lat, lng in cities:
        print(f"  Fetching {name} ({lat:.2f}, {lng:.2f}) ...", end=' ', flush=True)
        row = fetch_station(lat, lng)
        if row:
            row['city'] = name
            rows.append(row)
            pollutants_present = [k for k in row if k in ('PM2.5','PM10','NO2','SO2','OZONE','NH3','CO')]
            print(f"OK  [{', '.join(pollutants_present)}]")
        else:
            print("No data")
        time.sleep(0.5)   # be polite to the API
    return pd.DataFrame(rows)


def append_to_csv(new_df):
    cols = ['timestamp', 'city', 'station_name', 'lat', 'lng',
            'PM2.5', 'PM10', 'NO2', 'SO2', 'OZONE', 'NH3', 'CO']
    # Ensure all columns exist
    for c in cols:
        if c not in new_df.columns:
            new_df[c] = None
    new_df = new_df[cols]

    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(OUT_FILE, index=False)
    print(f"\n  Appended {len(new_df)} rows → {OUT_FILE}  (total: {len(combined)})")
    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch live AQI data from OpenAQ')
    parser.add_argument('--cities', nargs='+', metavar='CITY',
                        help='City names to filter from default list')
    parser.add_argument('--lat',  type=float, help='Single point latitude')
    parser.add_argument('--lng',  type=float, help='Single point longitude')
    args = parser.parse_args()

    print("=" * 50)
    print("  OpenAQ Historical Data Collector")
    print("=" * 50)

    if args.lat and args.lng:
        cities = [('custom', args.lat, args.lng)]
    elif args.cities:
        names_upper = [c.lower() for c in args.cities]
        cities = [(n, la, lo) for n, la, lo in DEFAULT_CITIES if n.lower() in names_upper]
        if not cities:
            print(f"No matching cities found. Available: {[n for n,_,_ in DEFAULT_CITIES]}")
            exit(1)
    else:
        cities = DEFAULT_CITIES

    print(f"  Fetching {len(cities)} location(s)...\n")
    df = fetch_all(cities)

    if not df.empty:
        append_to_csv(df)
    else:
        print("  No data retrieved.")
