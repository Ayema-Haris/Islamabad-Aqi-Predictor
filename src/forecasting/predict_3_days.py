"""
predict_3_days.py
──────────────────
Runs immediately after training (GitHub Actions: needs train-and-register).

What it does:
  1. Loads the latest registered model + scaler from Hopsworks Model Registry
  2. Fetches a 3-day (72-hour) weather/air-quality forecast from Open-Meteo
  3. Runs inference and saves:
       artifacts/predictions/next_72_hours.csv   – one row per hour
       artifacts/predictions/next_3_days.csv     – daily summary
       artifacts/predictions/forecast_summary.json – Streamlit-ready

Usage:
    python src/forecasting/predict_3_days.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
import hopsworks

from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

# ── Constants ──────────────────────────────────────────────────────────────
HOPSWORKS_HOST    = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "Islamabad_Aqi_Predictor"
MODEL_NAME        = "islamabad_aqi_model"

PROJECT_ROOT = Path(__file__).parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PRED_DIR     = ARTIFACT_DIR / "predictions"

LAT, LON = 33.6844, 73.0479

FEATURE_COLS = [
    "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "ozone", "sulphur_dioxide",
    "hour", "day", "month",
    "pm2_5_change", "pm10_change",
]


# ── AQI helpers ────────────────────────────────────────────────────────────

def pm25_to_aqi(pm: float) -> int:
    """US-EPA AQI from PM2.5 (µg/m³)."""
    bp = [
        (0.0,   12.0,   0,  50),
        (12.1,  35.4,  51, 100),
        (35.5,  55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for lo, hi, alo, ahi in bp:
        if lo <= pm <= hi:
            return round(((ahi - alo) / (hi - lo)) * (pm - lo) + alo)
    return 500


def aqi_label(aqi: int) -> tuple[str, str]:
    if aqi <= 50:   return "Good",                          "🟢"
    if aqi <= 100:  return "Moderate",                     "🟡"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups","🟠"
    if aqi <= 200:  return "Unhealthy",                    "🔴"
    if aqi <= 300:  return "Very Unhealthy",               "🟣"
    return "Hazardous",                                     "🟤"


# ── Data fetching ──────────────────────────────────────────────────────────

def fetch_3day_forecast() -> pd.DataFrame:
    """Fetch 72-hour hourly forecast from Open-Meteo Air Quality API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":  LAT,
        "longitude": LON,
        "hourly": [
            "pm2_5", "pm10", "carbon_monoxide",
            "nitrogen_dioxide", "ozone", "sulphur_dioxide",
        ],
        "timezone":      "Asia/Karachi",
        "forecast_days": 3,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)

    df["hour"]  = df["timestamp"].dt.hour.astype("int32")
    df["day"]   = df["timestamp"].dt.day.astype("int32")
    df["month"] = df["timestamp"].dt.month.astype("int32")

    df["pm2_5_change"] = df["pm2_5"].diff().fillna(0.0)
    df["pm10_change"]  = df["pm10"].diff().fillna(0.0)

    pollutants = ["pm2_5","pm10","carbon_monoxide",
                  "nitrogen_dioxide","ozone","sulphur_dioxide"]
    df[pollutants] = df[pollutants].ffill().fillna(0.0)

    return df


# ── Model loading ──────────────────────────────────────────────────────────

def load_model_and_scaler(mr):
    """Load best model + scaler from Hopsworks registry (fallback: local)."""
    try:
        hw_model   = mr.get_model(name=MODEL_NAME, version=None)
        model_dir  = Path(hw_model.download())
        model  = joblib.load(model_dir / "best_model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        print(f"  ✔  Loaded from registry  (version {hw_model.version})")
    except Exception as exc:
        print(f"  ⚠  Registry load failed ({exc}). Trying local artifacts ...")
        model  = joblib.load(ARTIFACT_DIR / "best_model.pkl")
        scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
        print("  ✔  Loaded from local artifacts.")
    return model, scaler


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print(f"  AQI Inference Pipeline  |  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # 1. Connect
    print("\n[1/5] Connecting to Hopsworks ...")
    project = hopsworks.login(
        host          = HOPSWORKS_HOST,
        project       = HOPSWORKS_PROJECT,
        api_key_value = os.getenv("HOPSWORKS_API_KEY"),
    )
    mr = project.get_model_registry()
    print(f"  ✔  Connected → {project.name}")

    # 2. Load model from registry
    print("\n[2/5] Loading model from registry ...")
    model, scaler = load_model_and_scaler(mr)

    # 3. Fetch forecast
    print("\n[3/5] Fetching 3-day forecast ...")
    df = fetch_3day_forecast()
    print(f"  ✔  {len(df)} hourly rows  |  "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    # 4. Predict
    print("\n[4/5] Running inference ...")
    X     = df[FEATURE_COLS].fillna(0).values
    X_sc  = scaler.transform(X)
    pm25_pred = np.clip(model.predict(X_sc), 0, None)

    df["predicted_pm2_5"] = pm25_pred.round(2)
    df["predicted_aqi"]   = df["predicted_pm2_5"].apply(pm25_to_aqi)

    # 5. Save predictions
    print("\n[5/5] Saving predictions ...")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Hourly CSV
    hourly_out = df[["timestamp", "predicted_pm2_5", "predicted_aqi"]].copy()
    hourly_out.to_csv(PRED_DIR / "next_72_hours.csv", index=False)

    # Daily summary CSV
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date").agg(
        avg_pm2_5 = ("predicted_pm2_5", "mean"),
        min_pm2_5 = ("predicted_pm2_5", "min"),
        max_pm2_5 = ("predicted_pm2_5", "max"),
        avg_aqi   = ("predicted_aqi",   "mean"),
        min_aqi   = ("predicted_aqi",   "min"),
        max_aqi   = ("predicted_aqi",   "max"),
    ).reset_index()
    daily = daily.round({"avg_pm2_5": 2, "avg_aqi": 0, "min_aqi": 0, "max_aqi": 0})
    daily["avg_aqi"] = daily["avg_aqi"].astype(int)
    daily["min_aqi"] = daily["min_aqi"].astype(int)
    daily["max_aqi"] = daily["max_aqi"].astype(int)
    daily.to_csv(PRED_DIR / "next_3_days.csv", index=False)

    # JSON summary for Streamlit
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "forecast_days": [],
    }
    for _, row in daily.iterrows():
        label, emoji = aqi_label(int(row["avg_aqi"]))
        summary["forecast_days"].append({
            "date":     str(row["date"]),
            "avg_aqi":  int(row["avg_aqi"]),
            "min_aqi":  int(row["min_aqi"]),
            "max_aqi":  int(row["max_aqi"]),
            "avg_pm2_5": float(row["avg_pm2_5"]),
            "category": label,
            "emoji":    emoji,
        })
    (PRED_DIR / "forecast_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  ✔  Saved to {PRED_DIR}/")

    # Console preview
    print("\n  📅  3-DAY FORECAST PREVIEW")
    print("  " + "-" * 40)
    for day in summary["forecast_days"]:
        print(f"  {day['emoji']}  {day['date']}  |  AQI {day['avg_aqi']}  "
              f"({day['category']})  |  PM2.5 {day['avg_pm2_5']:.1f} µg/m³")

    print("\n✅  Inference pipeline finished.\n")


if __name__ == "__main__":
    run()