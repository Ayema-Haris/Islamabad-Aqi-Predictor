# create_feature_view.py  — place in project root and run once
import os
import hopsworks
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

project = hopsworks.login(
    host          = "eu-west.cloud.hopsworks.ai",
    project       = "Islamabad_Aqi_Predictor",
    api_key_value = os.getenv("HOPSWORKS_API_KEY"),
)
fs = project.get_feature_store()

# Get the existing feature group
fg = fs.get_feature_group(name="islamabad_hourly_aqi", version=1)
print(f"Feature group found: {fg.name} v{fg.version} ({len(fg.read())} rows)")

# Create the feature view
fv = fs.create_feature_view(
    name        = "aqi_feature_view",
    version     = 1,
    query       = fg.select_all(),
    description = "Hourly AQI features for Islamabad",
)
print(f"Feature view created: {fv.name} v{fv.version}")
print("Done — check Hopsworks UI > Feature Store > Feature Views")