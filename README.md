@"
# Islamabad AQI Predictor

ML-powered 3-day air quality forecast for Islamabad using XGBoost, LightGBM, and CatBoost.

## Features
- Hourly data fetching from Open-Meteo API
- Automated ML model training with Hopsworks Model Registry
- 72-hour AQI predictions
- Interactive Streamlit dashboard
- GitHub Actions automation

## Local Setup
\`\`\`bash
pip install -r requirements.txt
python src/forecasting/predict_3_days.py
streamlit run src/app/streamlit_app.py
\`\`\`

## Architecture
- **Data**: Open-Meteo API → Hopsworks Feature Store
- **Training**: Daily (01:00 UTC) → Best model to Hopsworks Registry
- **Predictions**: Model from registry → Predictions committed to Git
- **Dashboard**: Streamlit Cloud reads from Git
"@ | Out-File -FilePath README.md -Encoding utf8