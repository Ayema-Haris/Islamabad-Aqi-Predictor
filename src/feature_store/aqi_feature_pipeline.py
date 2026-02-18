"""
aqi_feature_pipeline.py
────────────────────────
Runs every hour (GitHub Actions cron: '0 * * * *').

What it does:
  1. Fetches the latest hourly air-quality data from Open-Meteo
  2. Inserts new rows into the Hopsworks feature group `islamabad_hourly_aqi`
     (primary key = timestamp → duplicates are automatically handled by Hudi)
  3. On first ever run it also creates the feature view used by training

Usage:
    python src/feature_store/aqi_feature_pipeline.py
"""

import os
import sys
import hopsworks
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path so we can import fetch_open_meteo
sys.path.insert(0, str(Path(__file__).parent))
from fetch_open_meteo import fetch_latest_hour

load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

# ── Constants ──────────────────────────────────────────────────────────────
HOPSWORKS_HOST    = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "Islamabad_Aqi_Predictor"

FG_NAME    = "islamabad_hourly_aqi"
FG_VERSION = 1
FV_NAME    = "aqi_feature_view"
FV_VERSION = 1


def get_or_create_feature_view(fs, fg):
    """Return existing feature view or create it if it doesn't exist yet."""
    try:
        fv = fs.get_feature_view(name=FV_NAME, version=FV_VERSION)
        print(f"  ✔  Feature view '{FV_NAME}' v{FV_VERSION} already exists.")
    except Exception:
        print(f"  ⚙  Creating feature view '{FV_NAME}' v{FV_VERSION} ...")
        query = fg.select_all()
        fv = fs.create_feature_view(
            name=FV_NAME,
            version=FV_VERSION,
            query=query,
            description="Hourly AQI features for Islamabad – used for training & inference",
        )
        print(f"  ✔  Feature view created.")
    return fv


def run():
    print("=" * 60)
    print(f"  AQI Feature Pipeline  |  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # ── 1. Connect ─────────────────────────────────────────────────────────
    print("\n[1/4] Connecting to Hopsworks ...")
    project = hopsworks.login(
        host          = HOPSWORKS_HOST,
        project       = HOPSWORKS_PROJECT,
        api_key_value = os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()
    print(f"  ✔  Connected  →  project: {project.name}")

    # ── 2. Fetch data ──────────────────────────────────────────────────────
    print("\n[2/4] Fetching latest hourly data from Open-Meteo ...")
    df = fetch_latest_hour()
    if df.empty:
        print("  ⚠  No data returned – skipping insert.")
        return
    print(f"  ✔  {len(df)} rows fetched  |  latest: {df['timestamp'].max()}")

    # ── 3. Insert into feature group ───────────────────────────────────────
    print(f"\n[3/4] Inserting into '{FG_NAME}' v{FG_VERSION} ...")
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    # wait_for_job=False keeps the hourly pipeline fast;
    # Hudi deduplicates on primary key (timestamp) automatically
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"  ✔  Insert submitted  ({len(df)} rows)")

    # ── 4. Ensure feature view exists ──────────────────────────────────────
    print(f"\n[4/4] Checking feature view ...")
    get_or_create_feature_view(fs, fg)

    print("\n✅  Feature pipeline finished successfully.\n")


if __name__ == "__main__":
    run()