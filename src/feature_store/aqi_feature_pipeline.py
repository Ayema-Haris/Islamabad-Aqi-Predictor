# aqi_feature_pipeline.py
#Reads raw data from Hopsworks.
#Performs feature engineering (hour/day/month, pollutant changes, rolling averages, etc.).
#nserts engineered features into a new feature group in Hopsworks.

import pandas as pd
import hopsworks
from fetch_open_meteo import fetch_air_quality
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
import os
import sys

API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not API_KEY:
    # fallback for local testing
    if len(sys.argv) > 1:
        API_KEY = sys.argv[1]
    else:
        raise ValueError(
            "HOPSWORKS_API_KEY not set. Use environment variable or pass as CLI argument."
        )

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
PROJECT_NAME = "Islamabad_Aqi_Predictor"
FEATURE_GROUP_NAME = "islamabad_hourly_aqi"

# 3 months of historical data
START_DATE = "2025-11-02"
END_DATE = "2026-02-02"

# -----------------------------
# LOGIN TO HOPSWORKS
# -----------------------------
print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=API_KEY, host="eu-west.cloud.hopsworks.ai")

fs = project.get_feature_store()
print(f"Logged in to project: {project.name}")
print(f"Feature Store: {fs.name}")

# -----------------------------
# DELETE OLD FEATURE GROUP (if exists)
# -----------------------------
try:
    old_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=1)
    if old_fg:
        old_fg.delete()
        print("Old feature group deleted.")
except:
    pass

# -----------------------------
# CREATE NEW FEATURE GROUP
# -----------------------------
aqi_fg = fs.create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=1,
    description="Hourly AQI and weather data for Islamabad with time-based and derived features",
    primary_key=["timestamp"],
    online_enabled=False
)
print("New feature group created successfully.")

# -----------------------------
# LOOP OVER DATE RANGE AND PROCESS DATA
# -----------------------------
start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
current_dt = start_dt

while current_dt <= end_dt:
    next_dt = current_dt + timedelta(days=1)
    
    # Fetch raw data from API
    df = fetch_air_quality(current_dt.strftime("%Y-%m-%d"), next_dt.strftime("%Y-%m-%d"))
    
    if df.empty:
        print(f"No data for {current_dt.strftime('%Y-%m-%d')}, skipping.")
        current_dt = next_dt
        continue
    
    # -----------------------------
    # Convert timestamp to datetime
    # -----------------------------
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
    
    # -----------------------------
    # Sort by timestamp
    # -----------------------------
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # -----------------------------
    # Compute time-based features
    # -----------------------------
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    
    # -----------------------------
    # Compute derived features (change rates)
    # -----------------------------
    df['pm2_5_change'] = df['pm2_5'].diff().fillna(0)
    df['pm10_change'] = df['pm10'].diff().fillna(0)
    
    # -----------------------------
    # Insert into Hopsworks Feature Store
    # -----------------------------
    aqi_fg.insert(df, write_options={"mode": "append"})
    print(f"Data for {current_dt.strftime('%Y-%m-%d')} inserted/appended to feature group.")
    
    current_dt = next_dt

print("Feature pipeline completed successfully!")
