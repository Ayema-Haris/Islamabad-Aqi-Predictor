import os
import pandas as pd
import hopsworks
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

HOPSWORKS_HOST    = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "Islamabad_Aqi_Predictor"
FG_NAME    = "islamabad_hourly_aqi"
FG_VERSION = 1
FV_NAME    = "aqi_feature_view"
FV_VERSION = 1

print("\n[1/4] Connecting to Hopsworks ...")
api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise ValueError("HOPSWORKS_API_KEY not set. Run: $env:HOPSWORKS_API_KEY = 'your_key'")

project = hopsworks.login(
    host          = HOPSWORKS_HOST,
    project       = HOPSWORKS_PROJECT,
    api_key_value = api_key,
)
fs = project.get_feature_store()
print(f"  Connected to {project.name}")

print(f"\n[2/4] Reading feature group '{FG_NAME}' ...")
fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
df = fg.read()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"  Total rows    : {len(df):,}")
print(f"  Earliest      : {df['timestamp'].min()}")
print(f"  Latest        : {df['timestamp'].max()}")

READY = len(df) >= 100
if len(df) < 100:
    print(f"  NOT READY — only {len(df)} rows. Uncomment BACKFILL block.")
elif len(df) < 500:
    print(f"  WARNING — {len(df)} rows, limited but trainable.")
else:
    print(f"  READY — {len(df):,} rows.")

# BACKFILL — uncomment if you have fewer than 500 rows
# import sys; sys.path.insert(0, "src/feature_store")
# from fetch_open_meteo import fetch_air_quality
# end   = pd.Timestamp.now().strftime("%Y-%m-%d")
# start = (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
# df_hist = fetch_air_quality(start, end)
# fg.insert(df_hist, write_options={"wait_for_job": True})
# print(f"  Backfill done: {len(df_hist)} rows inserted.")

print(f"\n[3/4] Setting up feature view ...")
try:
    fv = fs.get_feature_view(name=FV_NAME, version=FV_VERSION)
    print(f"  Feature view already exists.")
except Exception:
    fv = fs.create_feature_view(
        name=FV_NAME, version=FV_VERSION,
        query=fg.select_all(),
        description="Hourly AQI features for Islamabad",
    )
    print(f"  Feature view created.")

print(f"\n[4/4] Summary:")
print(f"  Feature group : {FG_NAME} v{FG_VERSION} ({len(df):,} rows)")
print(f"  Feature view  : {FV_NAME} v{FV_VERSION}")
if READY:
    print(f"\n  Ready! Run: python src/model/train_models.py")
else:
    print(f"\n  Not ready. Uncomment BACKFILL block and re-run.")