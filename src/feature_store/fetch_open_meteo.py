import requests
import pandas as pd

LAT = 33.6844     # Islamabad
LON = 73.0479

def fetch_air_quality(start_date, end_date):
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
            "sulphur_dioxide"
        ]
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()

    data = resp.json()["hourly"]
    df = pd.DataFrame(data)

    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)

    return df
