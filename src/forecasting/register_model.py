import joblib
import hopsworks
import os

# Path to the model to register
MODEL_FILE = "artifacts/models/best_model.pkl"
MODEL_NAME = "aqi_predictor"

# Ensure the model exists
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found!")

# Connect to Hopsworks using API key from env
api_key = os.environ["HOPSWORKS_API_KEY"]
project = hopsworks.login(api_key_value=api_key)
mr = project.get_model_registry()

# Create model in registry (if it doesn't exist)
try:
    model = mr.python.get_model(MODEL_NAME)
except hopsworks.client.exceptions.ResourceNotFound:
    model = mr.python.create_model(
        name=MODEL_NAME,
        description="72-hour AQI prediction model",
        metrics={"rmse": 2.5},  # optional, adjust if you have metrics
        hyperparameters={"learning_rate": 0.01, "n_estimators": 100},  # optional
    )

# Add a new version
model_version = model.add_version(MODEL_FILE)
print(f"Registered model '{MODEL_NAME}', version {model_version.version}")
