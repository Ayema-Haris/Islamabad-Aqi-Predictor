import hopsworks

#Connects to Hopsworks and creates the raw feature group.
#Inserts raw features into Hopsworks.

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

# -----------------------------
# LOGIN
# -----------------------------
print("Connecting to Hopsworks...")

try:
    project = hopsworks.login(api_key_value=API_KEY, host="eu-west.cloud.hopsworks.ai")

    print("SUCCESS")
    print("Project name:", project.name)

    # Access feature store
    fs = project.get_feature_store()
    print("Feature Store:", fs.name)

except Exception as e:
    print("Error connecting to Hopsworks:", e)
