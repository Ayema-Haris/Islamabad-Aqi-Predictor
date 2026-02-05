# src/model_training/train_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import hopsworks

import os

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/metrics", exist_ok=True)

# -----------------------
# Connect to Hopsworks Feature Store
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


project = hopsworks.login(api_key_value=API_KEY, host="eu-west.cloud.hopsworks.ai")

fs = project.get_feature_store()
print("Connected to Hopsworks project:", project.name)

# -----------------------
# Load engineered features
# -----------------------
fg = fs.get_feature_group(name="islamabad_hourly_aqi_engineered", version=1)
df = fg.read()
print("Engineered features fetched, sample:")
print(df.head())

# -----------------------
# Prepare data
# -----------------------
target = "pm2_5"
features = [col for col in df.columns if col not in ["timestamp", target]]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # keep time order

# -----------------------
# Define models
# -----------------------
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "RidgeRegression": Ridge(alpha=1.0),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = {}

# -----------------------
# Train and evaluate
# -----------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    results[name] = {"model": model, "RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

# -----------------------
# Select best model (lowest RMSE)
# -----------------------
best_model_name = min(results, key=lambda x: results[x]["RMSE"])
best_model = results[best_model_name]["model"]

print(f"Best model: {best_model_name}")




# -----------------------
# Optional: save metrics for dashboard
# -----------------------
joblib.dump(best_model, "artifacts/models/best_model.pkl")
metrics_df = pd.DataFrame.from_dict(results, orient="index")
metrics_df.to_csv("artifacts/metrics/model_metrics.csv", index=False)

print("Best model saved as 'best_model.pkl'")

print("Metrics saved to 'model_metrics.csv'")
