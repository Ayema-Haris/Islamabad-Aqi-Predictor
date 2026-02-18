"""
train_models.py
────────────────
Runs once per day (GitHub Actions cron: '0 1 * * *').

What it does:
  1. Pulls training data from the Hopsworks feature view
  2. Trains XGBoost, LightGBM and CatBoost with time-series cross-validation
  3. Picks the best model by test RMSE (with an overfitting penalty)
  4. Saves model + scaler locally under artifacts/
  5. Registers the best model in the Hopsworks Model Registry
     - if a better model was already registered, it is NOT overwritten

Usage:
    python src/model/train_models.py
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import hopsworks

from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Load .env from project root (works whether you run from root or src/model/)
load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)

# ── Constants ──────────────────────────────────────────────────────────────
HOPSWORKS_HOST    = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "Islamabad_Aqi_Predictor"

FV_NAME      = "aqi_feature_view"
FV_VERSION   = 1
MODEL_NAME   = "islamabad_aqi_model"

# Always relative to project root regardless of where script is called from
PROJECT_ROOT = Path(__file__).parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

TARGET_COL   = "pm2_5"

FEATURE_COLS = [
    "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "ozone", "sulphur_dioxide",
    "hour", "day", "month",
    "pm2_5_change", "pm10_change",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def make_models() -> dict:
    """Return three regularised regressors."""
    return {
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=5,
            l2_leaf_reg=3.0, subsample=0.8,
            random_state=42, verbose=False,
        ),
    }


def evaluate(model, X_tr, X_te, y_tr, y_te) -> dict:
    """Train + evaluate; return metrics dict including overfitting flag."""
    tscv = TimeSeriesSplit(n_splits=5)
    cv   = cross_val_score(
        model, X_tr, y_tr, cv=tscv,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    cv_rmse = float(np.sqrt(-cv.mean()))
    cv_std  = float(np.sqrt(cv.std()))

    model.fit(X_tr, y_tr)

    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    train_rmse = float(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
    test_rmse  = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))
    test_mae   = float(mean_absolute_error(y_te, y_te_pred))
    test_r2    = float(r2_score(y_te, y_te_pred))

    ratio   = train_rmse / test_rmse if test_rmse > 0 else 1.0
    overfit = ratio < 0.80

    return {
        "cv_rmse": cv_rmse, "cv_std": cv_std,
        "train_rmse": train_rmse, "test_rmse": test_rmse,
        "test_mae": test_mae, "test_r2": test_r2,
        "overfitting_ratio": float(ratio), "is_overfitting": overfit,
    }


def load_training_data(fs):
    """Pull data directly from feature group, chronological 80/20 split."""
    print("  Loading from feature group ...")
    fg = fs.get_feature_group(name="islamabad_hourly_aqi", version=1)
    df = fg.read()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df.dropna(subset=[TARGET_COL], inplace=True)

    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL].values

    split = int(len(df) * 0.8)
    return X.iloc[:split], X.iloc[split:], y[:split], y[split:]


def register_model(mr, model, scaler, metrics: dict, model_name: str):
    """
    Save artifacts and register in Hopsworks Model Registry.
    Only registers if new model beats the currently registered one.
    """
    # Save to artifacts/ — predict_3_days.py reads from here too
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    model_path  = ARTIFACT_DIR / "best_model.pkl"
    scaler_path = ARTIFACT_DIR / "scaler.pkl"
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  ✔  Saved locally → {model_path}")

    # Check if a better model is already registered
    try:
        existing      = mr.get_model(name=MODEL_NAME, version=None)
        existing_rmse = existing.training_metrics.get("test_rmse", float("inf"))
        if metrics["test_rmse"] >= existing_rmse:
            print(f"  ⚠  New RMSE ({metrics['test_rmse']:.4f}) is not better than "
                  f"registered RMSE ({existing_rmse:.4f}). Skipping registration.")
            return None
        print(f"  ✔  New model beats existing "
              f"({metrics['test_rmse']:.4f} < {existing_rmse:.4f})")
    except Exception:
        print("  ✔  No existing model — registering for the first time.")

    hw_model = mr.python.create_model(
        name        = MODEL_NAME,
        description = f"{model_name} trained {datetime.now(timezone.utc).date()}",
        metrics     = {
            "test_rmse": round(metrics["test_rmse"], 4),
            "test_mae":  round(metrics["test_mae"],  4),
            "test_r2":   round(metrics["test_r2"],   4),
            "cv_rmse":   round(metrics["cv_rmse"],   4),
        },
        input_example = pd.DataFrame(
            [[0.0] * len(FEATURE_COLS)], columns=FEATURE_COLS
        ),
    )
    # Upload the whole artifacts/ folder (includes best_model.pkl + scaler.pkl)
    hw_model.save(str(ARTIFACT_DIR))
    print(f"  ✔  Registered as '{MODEL_NAME}' version {hw_model.version}")
    return hw_model


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print(f"  AQI Model Training  |  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # 1. Connect
    print("\n[1/5] Connecting to Hopsworks ...")
    project = hopsworks.login(
        host          = HOPSWORKS_HOST,
        project       = HOPSWORKS_PROJECT,
        api_key_value = os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    print(f"  ✔  Connected → {project.name}")

    # 2. Load data
    print("\n[2/5] Loading training data ...")
    X_tr, X_te, y_tr, y_te = load_training_data(fs)
    print(f"  ✔  Train: {len(X_tr)} rows  |  Test: {len(X_te)} rows")

    # 3. Scale
    print("\n[3/5] Scaling features ...")
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # 4. Train & evaluate all models
    print("\n[4/5] Training models ...")
    models  = make_models()
    results = {}
    best_name, best_score, best_model = None, float("inf"), None

    for name, mdl in models.items():
        print(f"\n  --- {name} ---")
        metrics = evaluate(mdl, X_tr_sc, X_te_sc, y_tr, y_te)
        results[name] = {"metrics": metrics, "model": mdl}

        print(f"  CV RMSE  : {metrics['cv_rmse']:.3f}  (+/- {metrics['cv_std']:.3f})")
        print(f"  Test RMSE: {metrics['test_rmse']:.3f}  |  R²: {metrics['test_r2']:.3f}")
        flag = "⚠  OVERFIT DETECTED" if metrics["is_overfitting"] else "✔  Good generalisation"
        print(f"  {flag}  (ratio={metrics['overfitting_ratio']:.3f})")

        adjusted = metrics["test_rmse"] * (1.2 if metrics["is_overfitting"] else 1.0)
        if adjusted < best_score:
            best_score = adjusted
            best_name  = name
            best_model = mdl

    print(f"\n  Best model: {best_name}  (adjusted score: {best_score:.3f})")

    # 5. Save & register
    print("\n[5/5] Saving & registering best model ...")
    best_metrics = results[best_name]["metrics"]

    metrics_summary = {
        "best_model":  best_name,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "all_results": {
            k: {m: v for m, v in r["metrics"].items() if m != "model"}
            for k, r in results.items()
        },
    }
    metrics_path = ARTIFACT_DIR / "metrics" / "model_metrics.json"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))

    register_model(mr, best_model, scaler, best_metrics, best_name)

    print("\n✅  Training pipeline finished.\n")


if __name__ == "__main__":
    run()