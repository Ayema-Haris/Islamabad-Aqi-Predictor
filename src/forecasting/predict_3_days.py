# src/forecasting/predict_next_3_days.py

import os
import sys
import joblib
import pandas as pd
import hopsworks

# -----------------------------
# 1. Setup paths
# -----------------------------
os.makedirs("artifacts/predictions", exist_ok=True)

# -----------------------------
# 2. Connect to Hopsworks
# -----------------------------
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        print("Error: HOPSWORKS_API_KEY not set. Provide as env var or first argument.")
        sys.exit(1)

print("Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=api_key, host="eu-west.cloud.hopsworks.ai")
print("Connected to Hopsworks successfully!")

# -----------------------------
# 3. Load latest model from Hopsworks Model Registry
# -----------------------------
MODEL_NAME = "aqi_predictor"
mr = project.get_model_registry()

try:
    model = mr.python.get_model(MODEL_NAME)
    latest_version = model.get_latest_version()
    model_file_path = latest_version.download()  # downloads locally
    model = joblib.load(model_file_path)
    print(f"Loaded model '{MODEL_NAME}', version {latest_version.version} from Hopsworks")
except hopsworks.client.exceptions.ResourceNotFound:
    print(f"Error: Model '{MODEL_NAME}' not found in Model Registry.")
    sys.exit(1)

# -----------------------------
# 4. Read feature data from Feature Store
# -----------------------------
fs = project.get_feature_store()
fg = fs.get_feature_group(name="islamabad_hourly_aqi_engineered", version=1)
df = fg.read()
print(f"Read {len(df)} rows from feature group '{fg.name}'")

# -----------------------------
# 5. Get latest known row
# -----------------------------
df = df.sort_values("timestamp")
latest = df.iloc[-1]
last_timestamp = latest["timestamp"]
print(f"Latest timestamp in data: {last_timestamp}")

# -----------------------------
# 6. Generate next 72 hours
# -----------------------------
future_times = pd.date_range(
    start=last_timestamp + pd.Timedelta(hours=1),
    periods=72,
    freq="H"
)
future_df = pd.DataFrame({"timestamp": future_times})

# -----------------------------
# 7. Build required features
# -----------------------------
future_df["hour"] = future_df["timestamp"].dt.hour
future_df["weekday"] = future_df["timestamp"].dt.weekday
future_df["day"] = future_df["timestamp"].dt.day
future_df["month"] = future_df["timestamp"].dt.month

# Fill all other features expected by the model
for col in model.feature_names_in_:
    if col not in future_df.columns:
        # Use latest known value if exists, otherwise 0
        future_df[col] = latest.get(col, 0)

# Ensure the columns are in the exact order the model expects
X_future = future_df[model.feature_names_in_]

# -----------------------------
# 8. Predict AQI
# -----------------------------
future_df["predicted_aqi"] = model.predict(X_future)
print("Predictions generated for the next 72 hours!")

# -----------------------------
# 9. Save predictions
# -----------------------------
output_file = "artifacts/predictions/next_72_hours.csv"
future_df.to_csv(output_file, index=False)
print(f"72-hour AQI forecast saved to {output_file}")

# Optional: preview first few predictions
print(future_df.head())
