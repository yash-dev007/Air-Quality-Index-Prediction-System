"""
Streamlit AQI Prediction App
==============================
Run from project root:
    streamlit run src/app.py
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from realtime_aqi_pipeline import (
    POLLUTANTS, AQI_CATEGORIES, INDIA_CITIES,
    predict_aqi, predict_aqi_with_interval, aqi_to_category, aqi_to_color,
    compute_sub_index, build_input_vector,
    get_city_station_stats,
    fetch_openaq_and_predict as _fetch_live,
    OUTPUT_DIR, FIGURES_DIR,
)
from config import WHO_GUIDELINES

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="India AQI Predictor",
    page_icon="🌿",
    layout="wide",
)

# ── Load model assets ─────────────────────────────────────────
MODEL_PATH    = os.path.join(OUTPUT_DIR, 'rt_best_model.pkl')
CLF_PATH      = os.path.join(OUTPUT_DIR, 'rt_classifier.pkl')
SHAP_PATH     = os.path.join(OUTPUT_DIR, 'rt_shap_explainer.pkl')
FEAT_PATH     = os.path.join(OUTPUT_DIR, 'rt_feature_names.pkl')
MEDIANS_PATH  = os.path.join(OUTPUT_DIR, 'rt_pollutant_medians.pkl')
STATIONS_PATH = os.path.join(OUTPUT_DIR, 'rt_stations_aqi.csv')
DRIFT_LOG     = os.path.join(OUTPUT_DIR, 'drift_log.csv')


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@st.cache_resource
def load_classifier():
    return joblib.load(CLF_PATH) if os.path.exists(CLF_PATH) else None

@st.cache_resource
def load_shap():
    return joblib.load(SHAP_PATH) if os.path.exists(SHAP_PATH) else None

@st.cache_data
def load_station_data():
    return pd.read_csv(STATIONS_PATH) if os.path.exists(STATIONS_PATH) else None

@st.cache_data
def load_medians():
    return joblib.load(MEDIANS_PATH) if os.path.exists(MEDIANS_PATH) else {}

@st.cache_data
def load_drift_log():
    if os.path.exists(DRIFT_LOG):
        df = pd.read_csv(DRIFT_LOG)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None


model      = load_model()
shap_exp   = load_shap()
station_df = load_station_data()
medians    = load_medians()


# ── Health advice lookup ──────────────────────────────────────
HEALTH_ADVICE = {
    'Good'        : ("✅ Air quality is good.", "No precautions needed. Enjoy outdoor activities freely.",
                     ["All groups can exercise outdoors.",
                      "Windows can be kept open.", "No mask required."]),
    'Satisfactory': ("🟡 Air quality is acceptable.", "Unusually sensitive people should take note.",
                     ["Sensitive individuals may experience mild symptoms.",
                      "Healthy adults can exercise outdoors normally.",
                      "Asthmatics should keep inhaler handy."]),
    'Moderate'    : ("🟠 Moderate air quality.", "Sensitive groups may experience health effects.",
                     ["Children, elderly, and those with heart/lung conditions should limit prolonged outdoor exertion.",
                      "Healthy adults can still exercise outdoors.",
                      "Consider wearing a mask if outside for long periods."]),
    'Poor'        : ("🔴 Poor air quality — take precautions.", "Everyone may start to feel effects.",
                     ["Avoid prolonged outdoor physical activity.",
                      "Sensitive groups should remain indoors.",
                      "Wear an N95 mask if going outdoors.",
                      "Keep windows closed and use air purifier indoors."]),
    'Very Poor'   : ("🚨 Very poor air quality — health alert!", "Everyone may experience serious effects.",
                     ["Avoid all outdoor exertion.",
                      "Sensitive groups must stay indoors.",
                      "Wear N95/FFP2 mask if outdoors.",
                      "Run air purifier with HEPA filter indoors.",
                      "Avoid opening windows."]),
    'Severe'      : ("☠️ Severe — Emergency conditions!", "Avoid all outdoor activity.",
                     ["Stay indoors with windows and doors sealed.",
                      "Use air purifier continuously.",
                      "Wear N95 mask even indoors if air purifier is unavailable.",
                      "Seek medical help if experiencing breathing difficulty.",
                      "Authorities may issue travel restrictions."]),
}

SENSITIVE_GROUPS = {
    'Good'        : [],
    'Satisfactory': ["Asthmatics", "Allergy sufferers"],
    'Moderate'    : ["Children", "Elderly", "Heart disease patients", "Lung disease patients", "Pregnant women"],
    'Poor'        : ["Everyone — but especially all above groups"],
    'Very Poor'   : ["Everyone"],
    'Severe'      : ["Everyone — emergency level"],
}


# ── UI helpers ────────────────────────────────────────────────
def aqi_badge(aqi, size="large"):
    color = aqi_to_color(aqi)
    cat   = aqi_to_category(aqi)
    fs = "2.4rem" if size == "large" else "1.4rem"
    return (
        f'<div style="background:{color};color:white;padding:14px 24px;'
        f'border-radius:12px;display:inline-block;text-align:center;">'
        f'<div style="font-size:{fs};font-weight:700;line-height:1">{aqi:.0f}</div>'
        f'<div style="font-size:0.9rem;margin-top:4px">{cat}</div>'
        f'</div>'
    )


def category_legend():
    cols = st.columns(len(AQI_CATEGORIES))
    for col, (lo, hi, label, color) in zip(cols, AQI_CATEGORIES):
        col.markdown(
            f'<div style="background:{color};color:white;border-radius:8px;'
            f'padding:6px;text-align:center;font-size:0.75rem;">'
            f'<b>{label}</b><br>{lo}–{hi}</div>',
            unsafe_allow_html=True,
        )


def health_panel(aqi):
    cat = aqi_to_category(aqi)
    color = aqi_to_color(aqi)
    title, summary, tips = HEALTH_ADVICE.get(cat, ("Unknown", "", []))
    sensitive = SENSITIVE_GROUPS.get(cat, [])

    st.markdown(
        f'<div style="border-left:5px solid {color};padding:12px 16px;'
        f'background:#f8f9fa;border-radius:0 8px 8px 0;margin:10px 0">'
        f'<b style="font-size:1.1rem">{title}</b><br>'
        f'<span style="color:#555">{summary}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if tips:
        st.markdown("**Recommendations:**")
        for tip in tips:
            st.markdown(f"- {tip}")
    if sensitive:
        st.markdown(f"**Sensitive groups:** {', '.join(sensitive)}")


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("🌿 India AQI Predictor")
st.sidebar.markdown("ML-powered air quality prediction with India NAQI.")

if model is None:
    st.sidebar.error("Model not found.\nRun: `python src/realtime_aqi_pipeline.py`")
    st.stop()

page = st.sidebar.radio("Navigate", [
    "Predict for Location",
    "Forecast",
    "City Comparison",
    "Station Map",
    "SHAP Explanation",
    "Alerts & Drift",
    "About",
])

# Optional: OpenWeatherMap API key in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Optional: Weather features**")
owm_key = st.sidebar.text_input("OpenWeatherMap API key", type="password",
                                  help="Get free key at openweathermap.org")
if not owm_key:
    try:
        owm_key = st.secrets.get("OWM_API_KEY", "")
    except Exception:
        owm_key = os.environ.get("OWM_API_KEY", "")

if st.sidebar.button("Reload model", help="Clear cache after retraining"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Data: India CPCB  |  Model: Gradient Boosting  |  AQI: India NAQI")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT FOR LOCATION
# ══════════════════════════════════════════════════════════════
if page == "Predict for Location":
    st.title("Predict AQI for Any Location")
    category_legend()
    st.markdown("---")

    col_loc, col_poll = st.columns([1, 1])

    with col_loc:
        st.subheader("Location")
        city_names  = ["Custom"] + [n for n, _, _ in INDIA_CITIES]
        city_choice = st.selectbox("Quick-select city", city_names)
        if city_choice != "Custom":
            _, preset_lat, preset_lng = next((n,la,lo) for n,la,lo in INDIA_CITIES
                                              if n == city_choice)
        else:
            preset_lat, preset_lng = 28.6139, 77.2090
        lat = st.number_input("Latitude",  value=float(preset_lat), format="%.4f",
                               min_value=6.0, max_value=37.0)
        lng = st.number_input("Longitude", value=float(preset_lng), format="%.4f",
                               min_value=66.0, max_value=98.0)

        st.subheader("Time")
        hour  = st.slider("Hour of day", 0, 23, datetime.datetime.now().hour)
        month = st.selectbox("Month", list(range(1, 13)),
                              index=datetime.datetime.now().month - 1,
                              format_func=lambda m: datetime.date(2000, m, 1).strftime('%B'))

    with col_poll:
        st.subheader("Pollutant Concentrations (µg/m³)")
        st.caption(f"Defaults are training medians. Adjust to reflect actual readings.")
        pollutant_vals = {}
        for p in POLLUTANTS:
            default_val = float(medians.get(p, 0.0))
            pollutant_vals[p] = st.number_input(p, 0.0, 2000.0, default_val, 1.0, key=f"p_{p}")

    st.markdown("---")

    if st.button("Predict AQI", type="primary", use_container_width=True):
        p_dict = {k: v for k, v in pollutant_vals.items() if v > 0}
        aqi, lower, upper, cat = predict_aqi_with_interval(
            lat, lng, p_dict if p_dict else None, hour=hour, month=month
        )

        # ── Main result ──────────────────────────────────────────
        res_col, detail_col = st.columns([1, 2])
        with res_col:
            st.markdown(aqi_badge(aqi), unsafe_allow_html=True)
            if lower is not None:
                st.caption(f"80% CI: {lower:.0f} – {upper:.0f}")

        with detail_col:
            st.markdown(f"**Location:** lat={lat:.4f}, lng={lng:.4f}")
            st.markdown(f"**Time:** {datetime.date(2000, month, 1).strftime('%B')}, {hour:02d}:00")

        # ── Pollutant breakdown chart ─────────────────────────────
        sub_indices = {p: compute_sub_index(v, p) for p, v in p_dict.items()
                       if not pd.isna(compute_sub_index(v, p))} if p_dict else {}
        if sub_indices:
            st.markdown("---")
            st.subheader("Pollutant Sub-Index Breakdown")
            import matplotlib.pyplot as plt
            dominant = max(sub_indices, key=sub_indices.get)
            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9fa')
            colors = ['#e74c3c' if p == dominant else '#3498db' for p in sub_indices]
            bars = ax.barh(list(sub_indices.keys()), list(sub_indices.values()),
                           color=colors, edgecolor='white', height=0.6)
            ax.axvline(aqi, color='black', lw=1.5, ls='--', label=f'Overall AQI={aqi:.0f}')
            for bnd in [50, 100, 200, 300, 400]:
                ax.axvline(bnd, color='gray', lw=0.6, ls=':', alpha=0.5)
            for bar, (p, v) in zip(bars, sub_indices.items()):
                ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                        f'{v:.0f}', va='center', fontsize=9)
            ax.set_xlabel('NAQI Sub-Index')
            ax.set_title(f'Dominant: {dominant} (sub-idx={sub_indices[dominant]:.0f})',
                         fontweight='bold')
            ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # ── WHO Guidelines comparison ─────────────────────────────
        who_hits = {p: v for p, v in p_dict.items()
                    if WHO_GUIDELINES.get(p) and v > WHO_GUIDELINES[p]}
        if who_hits:
            st.markdown("---")
            st.subheader("WHO 2021 Guidelines Comparison")
            who_rows = []
            for p, v in p_dict.items():
                guideline = WHO_GUIDELINES.get(p)
                if guideline:
                    who_rows.append({
                        'Pollutant': p,
                        'Measured (µg/m³)': f'{v:.1f}',
                        'WHO Guideline': f'{guideline}',
                        'Times over': f'{v/guideline:.1f}x',
                        'Exceeds': '⚠ YES' if v > guideline else 'OK',
                    })
            if who_rows:
                who_df = pd.DataFrame(who_rows)
                st.dataframe(who_df, hide_index=True, use_container_width=True)
                st.caption("WHO 2021 Annual Mean Guidelines (PM2.5: 5, PM10: 15, NO2: 10, SO2: 40, O3: 60 µg/m³)")

        # ── Multi-station aggregation ─────────────────────────────
        stats = get_city_station_stats(lat, lng)
        if stats and stats['n_stations'] > 1:
            st.markdown("---")
            st.subheader(f"Nearby Monitoring Stations ({stats['n_stations']} within 60 km)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean AQI",  f"{stats['mean_aqi']:.0f}")
            c2.metric("Min AQI",   f"{stats['min_aqi']:.0f}")
            c3.metric("Max AQI",   f"{stats['max_aqi']:.0f}")
            c4.metric("Std Dev",   f"{stats['std_aqi']:.0f}")
            with st.expander("Station details"):
                st.dataframe(pd.DataFrame(stats['stations']), hide_index=True,
                             use_container_width=True)

        # ── Health advice ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Health Advice")
        health_panel(aqi)

        # ── Weather context ────────────────────────────────────────
        if owm_key:
            st.markdown("---")
            st.subheader("Current Weather Context")
            from weather import fetch_weather, weather_impact_note
            with st.spinner("Fetching weather..."):
                weather = fetch_weather(lat, lng, owm_key)
            if weather:
                wc1, wc2, wc3, wc4 = st.columns(4)
                wc1.metric("Temperature", f"{weather['temp_c']:.1f} °C")
                wc2.metric("Humidity",    f"{weather['humidity']}%")
                wc3.metric("Wind Speed",  f"{weather['wind_speed']:.1f} m/s")
                wc4.metric("Pressure",    f"{weather['pressure']} hPa")
                note = weather_impact_note(weather)
                if note:
                    st.info(note)

    # Live OpenAQ fetch
    st.markdown("---")
    st.subheader("Fetch Live Data (OpenAQ)")
    if st.button("Get live readings + predict"):
        with st.spinner("Querying OpenAQ API..."):
            try:
                import requests
                result = _fetch_live(lat, lng)
                if result:
                    aqi_live, cat_live = result
                    st.success(f"Live prediction: AQI **{aqi_live:.0f}** ({cat_live})")
                    st.markdown(aqi_badge(aqi_live), unsafe_allow_html=True)
                    health_panel(aqi_live)
                else:
                    st.warning("No data found nearby. Try a larger city.")
            except Exception as e:
                st.error(f"API error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST
# ══════════════════════════════════════════════════════════════
elif page == "Forecast":
    st.title("AQI Forecast")
    category_legend()
    st.markdown("---")

    from forecast import hourly_forecast, monthly_forecast, predict_next_day

    city_names  = [n for n, _, _ in INDIA_CITIES]
    city_choice = st.selectbox("City", city_names, key="fc_city")
    _, fc_lat, fc_lng = next((n,la,lo) for n,la,lo in INDIA_CITIES if n == city_choice)

    tab1, tab2, tab3 = st.tabs(["Hourly (Today)", "Seasonal (Monthly)", "Next-Day (Lag Model)"])

    with tab1:
        fc_month = st.selectbox("Month", list(range(1, 13)),
                                 index=datetime.datetime.now().month - 1,
                                 format_func=lambda m: datetime.date(2000,m,1).strftime('%B'),
                                 key="fc_month")
        with st.spinner("Computing hourly forecast..."):
            hourly = hourly_forecast(fc_lat, fc_lng, month=fc_month)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 3.5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        colors = [aqi_to_color(v) for v in hourly['AQI']]
        ax.bar(hourly['hour'], hourly['AQI'], color=colors, edgecolor='white', width=0.8)
        ax.plot(hourly['hour'], hourly['AQI'], 'k-o', ms=4, lw=1.5, alpha=0.5)
        for bnd in [50, 100, 200, 300]:
            ax.axhline(bnd, color='gray', lw=0.7, ls=':', alpha=0.5)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{h:02d}h' for h in range(24)], fontsize=8)
        ax.set_ylabel('Predicted AQI')
        ax.set_title(f'Hourly AQI Forecast — {city_choice}', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Peak/low
        peak_h = hourly.loc[hourly['AQI'].idxmax()]
        low_h  = hourly.loc[hourly['AQI'].idxmin()]
        c1, c2 = st.columns(2)
        c1.metric("Peak AQI", f"{peak_h['AQI']:.0f} ({peak_h['Category']})",
                  f"at {int(peak_h['hour']):02d}:00")
        c2.metric("Best hour", f"{low_h['AQI']:.0f} ({low_h['Category']})",
                  f"at {int(low_h['hour']):02d}:00")

    with tab2:
        fc_hour = st.slider("Hour of day", 0, 23, 12, key="fc_hour2")
        with st.spinner("Computing seasonal forecast..."):
            monthly = monthly_forecast(fc_lat, fc_lng, hour=fc_hour)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9fa')
        colors = [aqi_to_color(v) for v in monthly['AQI']]
        ax.bar(monthly['month_name'], monthly['AQI'], color=colors, edgecolor='white', width=0.7)
        ax.plot(monthly['month_name'], monthly['AQI'], 'k-o', ms=5, lw=1.5, alpha=0.5)
        for bnd in [50, 100, 200, 300]:
            ax.axhline(bnd, color='gray', lw=0.7, ls=':', alpha=0.5)
        ax.set_ylabel('Predicted AQI')
        ax.set_title(f'Seasonal AQI Forecast — {city_choice}', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.dataframe(monthly[['month_name','AQI','Category']].rename(
            columns={'month_name':'Month'}), hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("Uses a lag-feature model trained on `data/historical_aqi.csv`.")
        st.info("Run `python src/fetch_historical.py` daily to accumulate historical data. "
                "The lag model needs at least 2 weeks of data to be meaningful.")
        with st.spinner("Running lag model..."):
            aqi_nd, cat_nd = predict_next_day(city_choice)
        if aqi_nd:
            st.markdown(aqi_badge(aqi_nd), unsafe_allow_html=True)
            st.markdown(f"**Predicted tomorrow's AQI: {aqi_nd:.0f} — {cat_nd}**")
            health_panel(aqi_nd)
        else:
            st.warning(cat_nd)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CITY COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "City Comparison":
    st.title("AQI Comparison — Top Indian Cities")
    category_legend()
    st.markdown("---")

    import matplotlib.pyplot as plt
    col1, col2 = st.columns(2)
    hour_c  = col1.slider("Hour of day", 0, 23, datetime.datetime.now().hour, key="cc_hour")
    month_c = col2.selectbox("Month", list(range(1, 13)),
                              index=datetime.datetime.now().month - 1,
                              format_func=lambda m: datetime.date(2000,m,1).strftime('%B'),
                              key="cc_month")

    with st.spinner("Computing predictions..."):
        rows = []
        for city, lat, lng in INDIA_CITIES:
            aqi, cat = predict_aqi(lat, lng, hour=hour_c, month=month_c)
            rows.append({'City': city, 'Predicted AQI': aqi, 'Category': cat,
                         'lat': lat, 'lng': lng})
        city_df = pd.DataFrame(rows).sort_values('Predicted AQI', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9fa')
    colors = [aqi_to_color(v) for v in city_df['Predicted AQI']]
    bars = ax.barh(city_df['City'][::-1], city_df['Predicted AQI'][::-1],
                   color=colors[::-1], edgecolor='white', height=0.7)
    for bar, row in zip(bars, city_df.iloc[::-1].itertuples()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{row._3:.0f}  {row.Category}", va='center', fontsize=9)
    for b in [50, 100, 200, 300, 400]:
        ax.axvline(b, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlim(0, city_df['Predicted AQI'].max() * 1.3)
    ax.set_title(f'Predicted AQI — {datetime.date(2000,month_c,1).strftime("%B")}, '
                 f'{hour_c:02d}:00', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted AQI')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(city_df[['City','Predicted AQI','Category']], hide_index=True, use_container_width=True)

    # Health summary for worst city
    worst = city_df.iloc[0]
    st.markdown(f"---\n**Health advice for worst city — {worst['City']} (AQI {worst['Predicted AQI']:.0f}):**")
    health_panel(worst['Predicted AQI'])


# ══════════════════════════════════════════════════════════════
# PAGE 4 — STATION MAP
# ══════════════════════════════════════════════════════════════
elif page == "Station Map":
    st.title("India CPCB Monitoring Stations — AQI Map")
    category_legend()
    st.markdown("---")

    if station_df is None:
        st.warning("Station data not found. Run the pipeline first.")
    else:
        cat_filter = st.multiselect(
            "Filter by category",
            [c for _, _, c, _ in AQI_CATEGORIES],
            default=[c for _, _, c, _ in AQI_CATEGORIES],
        )
        filtered = station_df[station_df['AQI_Category'].isin(cat_filter)] if cat_filter else station_df
        st.caption(f"Showing {len(filtered):,} of {len(station_df):,} stations")

        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=[22.5, 82.0], zoom_start=5, tiles='CartoDB positron')
            for _, row in filtered.iterrows():
                color = aqi_to_color(row['AQI'])
                popup_lines = [
                    f"<b>{row.get('station', row.get('city','Station'))}</b>",
                    f"City: {row.get('city','')}",
                    f"AQI: {row['AQI']:.0f} ({row['AQI_Category']})",
                ]
                for p in POLLUTANTS:
                    if p in row and pd.notna(row[p]):
                        popup_lines.append(f"{p}: {row[p]:.1f} µg/m³")
                folium.CircleMarker(
                    location=[row['lat'], row['lng']],
                    radius=7,
                    color=color, fill=True, fill_color=color, fill_opacity=0.85,
                    popup=folium.Popup("<br>".join(popup_lines), max_width=220),
                    tooltip=f"{row.get('city','')} — AQI {row['AQI']:.0f}",
                ).add_to(m)
            st_folium(m, width=900, height=600)

        except ImportError:
            st.error("Install folium: `pip install folium streamlit-folium`")
            map_df = filtered[['lat','lng']].rename(columns={'lat':'latitude','lng':'longitude'})
            st.map(map_df)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — SHAP EXPLANATION
# ══════════════════════════════════════════════════════════════
elif page == "SHAP Explanation":
    st.title("SHAP — Feature Attribution")
    st.markdown(
        "SHAP values show how much each feature **pushes the prediction higher or lower** "
        "from the model's average."
    )

    shap_bar = os.path.join(FIGURES_DIR, 'rt_fig9b_shap_bar.png')
    shap_sum = os.path.join(FIGURES_DIR, 'rt_fig9a_shap_summary.png')

    if os.path.exists(shap_bar) and os.path.exists(shap_sum):
        c1, c2 = st.columns(2)
        c1.image(shap_bar, caption="Mean |SHAP| — global feature importance", use_column_width=True)
        c2.image(shap_sum, caption="Beeswarm — direction & magnitude", use_column_width=True)
    else:
        st.warning("SHAP plots not found. Run the full pipeline.")

    st.markdown("---")
    st.subheader("Explain a single prediction")

    sc1, sc2 = st.columns(2)
    city_s = sc1.selectbox("City", [n for n,_,_ in INDIA_CITIES], key="shap_city")
    _, s_lat, s_lng = next((n,la,lo) for n,la,lo in INDIA_CITIES if n == city_s)
    lat_s = sc1.number_input("Latitude",  value=float(s_lat), format="%.4f", key="shap_lat")
    lng_s = sc2.number_input("Longitude", value=float(s_lng), format="%.4f", key="shap_lng")

    pol_s = {}
    for p in POLLUTANTS:
        pol_s[p] = st.number_input(p, 0.0, 2000.0, float(medians.get(p, 0.0)), 1.0, key=f"shap_{p}")

    if st.button("Explain this prediction", key="shap_btn"):
        if shap_exp is None:
            st.warning("SHAP explainer not found. Run the full pipeline.")
        else:
            try:
                import shap
                import matplotlib.pyplot as plt

                feat_names = joblib.load(FEAT_PATH)
                p_dict = {k: v for k, v in pol_s.items() if v > 0}
                X_single = build_input_vector(lat_s, lng_s, p_dict or None,
                                              feature_names=feat_names)
                aqi_val = float(model.predict(X_single)[0])
                aqi_val = max(0, round(aqi_val, 1))
                st.markdown(aqi_badge(aqi_val), unsafe_allow_html=True)
                health_panel(aqi_val)

                shap_vals = shap_exp.shap_values(X_single)
                fig, _ = plt.subplots(figsize=(8, 4))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_vals[0],
                        base_values=shap_exp.expected_value,
                        data=X_single.values[0],
                        feature_names=feat_names,
                    ),
                    show=False,
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"SHAP error: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 6 — ALERTS & DRIFT
# ══════════════════════════════════════════════════════════════
elif page == "Alerts & Drift":
    st.title("Alerts & Model Drift Tracking")
    tab_alert, tab_drift = st.tabs(["Email Alerts", "Model Drift"])

    with tab_alert:
        st.subheader("AQI Threshold Alert Configuration")
        st.markdown(
            "Configure `alerts_config.json` in the project root to enable email alerts. "
            "Run `python src/aqi_alert.py --dry-run` to test without sending email."
        )
        st.code("""{
  "smtp_host"    : "smtp.gmail.com",
  "smtp_port"    : 587,
  "smtp_user"    : "you@gmail.com",
  "smtp_password": "your_app_password",
  "to_email"     : "you@gmail.com",
  "threshold_aqi": 150,
  "cities": [
    {"name": "Delhi",  "lat": 28.6139, "lng": 77.2090},
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777}
  ]
}""", language="json")

        st.markdown("**Quick check — current predicted AQI vs threshold:**")
        threshold = st.slider("AQI threshold", 50, 400, 150, key="alert_thresh")
        import matplotlib.pyplot as plt
        rows = []
        for city, lat, lng in INDIA_CITIES[:10]:
            aqi, cat = predict_aqi(lat, lng)
            rows.append({'City': city, 'AQI': aqi, 'Category': cat, 'Alert': aqi >= threshold})
        alert_df = pd.DataFrame(rows)
        triggered = alert_df[alert_df['Alert']]

        if triggered.empty:
            st.success(f"No cities currently above AQI {threshold}.")
        else:
            st.error(f"{len(triggered)} city/cities above threshold {threshold}: "
                     f"{', '.join(triggered['City'].tolist())}")

        def color_alert(row):
            bg = aqi_to_color(row['AQI'])
            return [f'background-color:{bg};color:white' if row['Alert']
                    else '' for _ in row]
        st.dataframe(alert_df.style.apply(color_alert, axis=1),
                     hide_index=True, use_container_width=True)

        st.markdown("---")
        st.markdown("**Schedule daily via Windows Task Scheduler:**")
        st.code("python D:\\Projects\\Predicting Air Quality Index\\src\\aqi_alert.py")

    with tab_drift:
        st.subheader("Model Drift Report")
        drift_df = load_drift_log()

        if drift_df is None:
            st.info("No drift log yet. Run `python src/drift_tracker.py --log` to start tracking.")
            st.code("# Log today's predictions\npython src/drift_tracker.py --log\n\n"
                    "# Generate report\npython src/drift_tracker.py --report")
        else:
            df_actual = drift_df.dropna(subset=['actual_aqi'])
            st.markdown(f"**Logged predictions:** {len(drift_df)}  |  "
                        f"**With actuals:** {len(df_actual)}  |  "
                        f"**Cities:** {drift_df['city'].nunique()}")

            # Prediction history chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('white'); ax.set_facecolor('#f8f9fa')
            for city in drift_df['city'].unique():
                cdf = drift_df[drift_df['city'] == city].sort_values('timestamp')
                ax.plot(cdf['timestamp'], cdf['predicted_aqi'], marker='o',
                        ms=3, label=city, alpha=0.7)
            ax.set_title('Predicted AQI History', fontweight='bold')
            ax.set_ylabel('Predicted AQI')
            ax.legend(fontsize=7, ncol=5)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            if not df_actual.empty:
                mae  = df_actual['error'].abs().mean()
                bias = df_actual['error'].mean()
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE",  f"{mae:.2f}")
                m2.metric("Bias", f"{bias:+.2f}",
                          delta_color="inverse" if abs(bias) > 10 else "normal")
                m3.metric("Samples with actuals", len(df_actual))

                drift_img = os.path.join(FIGURES_DIR, 'rt_drift_report.png')
                if os.path.exists(drift_img):
                    st.image(drift_img, use_column_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About This Project")
    st.markdown("""
### India Real-Time AQI Prediction

Predicts the **India National Air Quality Index (NAQI)** using a Gradient Boosting
model trained on live CPCB monitoring station data.

---

#### Pipeline Architecture
| Step | Description |
|---|---|
| Data | 487 India CPCB stations, 7 pollutants, long-format CSV |
| AQI | India NAQI breakpoints: concentrations (µg/m³) -> sub-indices; overall = max |
| Features | Geographic + Temporal (hour/month/season) + Pollutant concentrations |
| Model | Gradient Boosting (Optuna-tunable), CV R² ≈ 0.97 |
| Classifier | Random Forest predicts AQI *category* directly |
| Explainability | SHAP TreeExplainer — per-prediction waterfall charts |
| Forecast | Hourly + seasonal forecast; next-day lag model from historical data |
| Alerts | Email alerts via `aqi_alert.py` when threshold exceeded |
| Drift | Prediction logging + drift detection via `drift_tracker.py` |

---

#### CLI Quick Reference
```bash
# Full training pipeline
python src/realtime_aqi_pipeline.py

# With Optuna hyperparameter tuning (~5 min)
python src/realtime_aqi_pipeline.py --tune

# Predict for a location
python src/realtime_aqi_pipeline.py --predict 28.6 77.2

# Fetch live from OpenAQ
python src/realtime_aqi_pipeline.py --live 28.6 77.2

# Collect historical data (run daily)
python src/fetch_historical.py

# Email alerts
python src/aqi_alert.py --dry-run

# Drift tracking
python src/drift_tracker.py --log --report

# Hourly/seasonal forecast plots
python src/forecast.py --lat 28.6 --lng 77.2 --plot
```

---

#### Deploying to Streamlit Cloud
1. Push to GitHub (exclude `venv/`, `data/`, `.streamlit/secrets.toml`)
2. Go to share.streamlit.io → New app → select repo → `src/app.py`
3. Add secrets in App Settings > Secrets (see `.streamlit/secrets.toml.example`)
""")

    for lo, hi, label, color in AQI_CATEGORIES:
        st.markdown(
            f'<span style="background:{color};color:white;padding:3px 12px;'
            f'border-radius:6px;margin-right:6px">{label} ({lo}–{hi})</span>',
            unsafe_allow_html=True,
        )
