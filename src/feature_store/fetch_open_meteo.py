import requests
import pandas as pd

LAT = 33.6844   # Islamabad
LON = 73.0479


def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly air quality data from Open-Meteo and return a DataFrame
    whose columns match the Hopsworks feature group `islamabad_hourly_aqi`
    exactly:
        pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, ozone,
        sulphur_dioxide, timestamp, hour, day, month,
        pm2_5_change, pm10_change

    Parameters
    ----------
    start_date : str  e.g. "2024-01-01"
    end_date   : str  e.g. "2024-01-31"
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
        ],
        "timezone": "Asia/Karachi",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)

    # ── 1. Rename & parse timestamp ───────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)

    # ── 2. Time-based features (must match FG schema: int columns) ────────
    df["hour"]  = df["timestamp"].dt.hour.astype("int32")
    df["day"]   = df["timestamp"].dt.day.astype("int32")
    df["month"] = df["timestamp"].dt.month.astype("int32")

    # ── 3. Derived lag features ───────────────────────────────────────────
    # hour-over-hour change — first row will be NaN, fill with 0
    df["pm2_5_change"] = df["pm2_5"].diff().fillna(0.0)
    df["pm10_change"]  = df["pm10"].diff().fillna(0.0)

    # ── 4. Drop any rows where ALL pollutants are NaN (API gaps) ─────────
    pollutants = ["pm2_5", "pm10", "carbon_monoxide",
                  "nitrogen_dioxide", "ozone", "sulphur_dioxide"]
    df.dropna(subset=pollutants, how="all", inplace=True)

    # ── 5. Fill remaining NaNs with forward-fill then 0 ──────────────────
    df[pollutants] = df[pollutants].ffill().fillna(0.0)

    # ── 6. Enforce correct dtypes for Hopsworks (double = float64) ────────
    for col in pollutants + ["pm2_5_change", "pm10_change"]:
        df[col] = df[col].astype("float64")

    # ── 7. Return only the 12 columns the FG expects, in order ───────────
    ordered_cols = [
        "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
        "ozone", "sulphur_dioxide",
        "timestamp", "hour", "day", "month",
        "pm2_5_change", "pm10_change",
    ]
    return df[ordered_cols].reset_index(drop=True)


def fetch_latest_hour() -> pd.DataFrame:
    """
    Convenience wrapper used by the hourly pipeline:
    fetches only today's date so we get the most recent rows.
    The feature pipeline will deduplicate on `timestamp` (primary key).
    """
    today = pd.Timestamp.now(tz="Asia/Karachi").strftime("%Y-%m-%d")
    return fetch_air_quality(today, today)


if __name__ == "__main__":
    # Quick smoke-test
    df = fetch_latest_hour()
    print(df.tail(3).to_string())
    print(f"\nShape: {df.shape}  |  dtypes:\n{df.dtypes}")