# src/feature_store/aqi_feature_engineering.py

import hopsworks
import pandas as pd

# -----------------------
# Hopsworks connection
# -----------------------
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

project = hopsworks.login(api_key_value=API_KEY, host="eu-west.cloud.hopsworks.ai")

fs = project.get_feature_store()
print("Connected to Hopsworks project:", project.name)

# -----------------------
# Read raw features
# -----------------------
fg_raw = fs.get_feature_group(name="islamabad_hourly_aqi", version=1)
df = fg_raw.read()
print("Raw features fetched, sample:")
print(df.head())

# -----------------------
# Feature Engineering
# -----------------------

# Convert timestamp to datetime if numeric
if pd.api.types.is_integer_dtype(df['timestamp']):
    df['timestamp'] = pd.to_datetime(df['timestamp'] // 1000, unit='ms')
elif pd.api.types.is_datetime64_any_dtype(df['timestamp']):
    df['timestamp'] = df['timestamp']  # already datetime
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Time-based features
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday

# Change-rate features (difference from previous hour)
df['pm2_5_change'] = df['pm2_5'].diff().fillna(0)
df['pm10_change'] = df['pm10'].diff().fillna(0)

# Rolling averages (3-hour window)
df['pm2_5_3h_avg'] = df['pm2_5'].rolling(window=3, min_periods=1).mean()
df['pm10_3h_avg'] = df['pm10'].rolling(window=3, min_periods=1).mean()

# -----------------------
# Create or get engineered feature group (offline only)
# -----------------------
# Try fetching existing offline feature group
fg_engineered = fs.get_feature_group(name="islamabad_hourly_aqi_engineered", version=1)

# If it returns None, create a new offline FG
if fg_engineered is None:
    fg_engineered = fs.create_feature_group(
        name="islamabad_hourly_aqi_engineered",
        version=1,
        description="Engineered AQI features with time-based and change-rate features",
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False   # offline only
    )
    print("Feature group created successfully (offline only).")
else:
    print("Existing feature group fetched successfully.")


# -----------------------
# Insert engineered features
# -----------------------
fg_engineered.insert(df, write_options={"mode": "overwrite"})
print("Engineered features inserted successfully.")
