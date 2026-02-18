"""
streamlit_app.py
─────────────────
Interactive dashboard for Islamabad AQI predictions.
Reads predictions from artifacts/predictions/ (updated daily by GitHub Actions).

Usage:
    streamlit run src/app/streamlit_app.py
"""

import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parents[2]
PRED_DIR     = PROJECT_ROOT / "artifacts" / "predictions"

def get_aqi_color(aqi: int) -> str:
    if aqi <= 50:   return "#00E400"
    if aqi <= 100:  return "#FFFF00"
    if aqi <= 150:  return "#FF7E00"
    if aqi <= 200:  return "#FF0000"
    if aqi <= 300:  return "#8F3F97"
    return "#7E0023"

def get_health_message(aqi: int) -> str:
    if aqi <= 50:
        return "✅ **Air quality is satisfactory.** Enjoy outdoor activities!"
    if aqi <= 100:
        return "⚠️ **Moderate air quality.** Unusually sensitive people should consider limiting prolonged outdoor exertion."
    if aqi <= 150:
        return "🟠 **Unhealthy for sensitive groups.** Children, elderly, and people with respiratory conditions should limit outdoor activities."
    if aqi <= 200:
        return "🔴 **Unhealthy.** Everyone may experience health effects. Sensitive groups should avoid outdoor exertion."
    if aqi <= 300:
        return "🟣 **Very unhealthy.** Health alert: everyone may experience serious health effects. Avoid outdoor activities."
    return "🟤 **Hazardous.** Health warnings of emergency conditions. Everyone should avoid outdoor exposure."

st.set_page_config(
    page_title="Islamabad AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .forecast-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌫️ Islamabad Air Quality Forecast</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">3-Day AQI Predictions powered by Machine Learning</p>', unsafe_allow_html=True)

try:
    summary_path = PRED_DIR / "forecast_summary.json"
    if not summary_path.exists():
        st.error("⚠️ No forecast data found. Run: `python src/forecasting/predict_3_days.py`")
        st.stop()

    with open(summary_path) as f:
        summary = json.load(f)

    hourly_path = PRED_DIR / "next_72_hours.csv"
    df_hourly = pd.read_csv(hourly_path)
    df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"])

    forecast_days = summary["forecast_days"]
    generated_at  = datetime.fromisoformat(summary["generated_at"])

except Exception as e:
    st.error(f"❌ Error loading forecast data: {e}")
    st.stop()

today = forecast_days[0]
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 6px solid {get_aqi_color(today['avg_aqi'])};">
        <div style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">Today's Average AQI</div>
        <div style="font-size: 4rem; font-weight: 800; color: {get_aqi_color(today['avg_aqi'])};">
            {today['emoji']} {today['avg_aqi']}
        </div>
        <div style="font-size: 1.4rem; margin-top: 0.5rem; color: #444;">
            {today['category']}
        </div>
        <div style="font-size: 0.9rem; color: #888; margin-top: 1rem;">
            PM2.5: {today['avg_pm2_5']:.1f} µg/m³
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🏥 Health Advisory")
st.info(get_health_message(today['avg_aqi']))

st.markdown("---")
st.markdown("### 📅 3-Day Forecast")

cols = st.columns(3)
for i, day in enumerate(forecast_days):
    with cols[i]:
        color = get_aqi_color(day['avg_aqi'])
        st.markdown(f"""
        <div class="forecast-card" style="border-left-color: {color};">
            <div style="font-size: 0.9rem; color: #666;">{day['date']}</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: {color}; margin: 0.5rem 0;">
                {day['emoji']} {day['avg_aqi']}
            </div>
            <div style="font-size: 1rem; color: #444; margin-bottom: 0.5rem;">
                {day['category']}
            </div>
            <div style="font-size: 0.85rem; color: #888;">
                Range: {day['min_aqi']} - {day['max_aqi']}<br>
                PM2.5: {day['avg_pm2_5']:.1f} µg/m³
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📊 72-Hour AQI Trend")

fig = go.Figure()

fig.add_hrect(y0=0,   y1=50,  fillcolor="#00E400", opacity=0.1, line_width=0)
fig.add_hrect(y0=50,  y1=100, fillcolor="#FFFF00", opacity=0.1, line_width=0)
fig.add_hrect(y0=100, y1=150, fillcolor="#FF7E00", opacity=0.1, line_width=0)
fig.add_hrect(y0=150, y1=200, fillcolor="#FF0000", opacity=0.1, line_width=0)
fig.add_hrect(y0=200, y1=300, fillcolor="#8F3F97", opacity=0.1, line_width=0)
fig.add_hrect(y0=300, y1=500, fillcolor="#7E0023", opacity=0.1, line_width=0)

fig.add_trace(go.Scatter(
    x=df_hourly["timestamp"],
    y=df_hourly["predicted_aqi"],
    mode="lines+markers",
    name="Predicted AQI",
    line=dict(color="#667eea", width=3),
    marker=dict(size=4),
    hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>AQI: %{y}<extra></extra>",
))

fig.update_layout(
    xaxis_title="Date & Time",
    yaxis_title="AQI",
    height=400,
    hovermode="x unified",
    plot_bgcolor="white",
    showlegend=False,
    margin=dict(l=20, r=20, t=20, b=20),
)

fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", range=[0, max(df_hourly["predicted_aqi"]) * 1.1])

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 🎨 AQI Scale Reference")

scale_cols = st.columns(6)
scale_data = [
    ("🟢", "Good", "0-50", "#00E400"),
    ("🟡", "Moderate", "51-100", "#FFFF00"),
    ("🟠", "Unhealthy for Sensitive", "101-150", "#FF7E00"),
    ("🔴", "Unhealthy", "151-200", "#FF0000"),
    ("🟣", "Very Unhealthy", "201-300", "#8F3F97"),
    ("🟤", "Hazardous", "301+", "#7E0023"),
]

for i, (emoji, label, range_val, color) in enumerate(scale_data):
    with scale_cols[i]:
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: {color}22; border-radius: 6px;">
            <div style="font-size: 1.5rem;">{emoji}</div>
            <div style="font-size: 0.75rem; font-weight: 600; margin-top: 0.25rem;">{label}</div>
            <div style="font-size: 0.7rem; color: #666;">{range_val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div class="footer">
    Last updated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}<br>
    Model from Hopsworks Registry | Predictions by XGBoost/LightGBM/CatBoost | Data from Open-Meteo API<br>
    <small>⚠️ For informational purposes only. Not a substitute for official air quality monitoring.</small>
</div>
""", unsafe_allow_html=True)