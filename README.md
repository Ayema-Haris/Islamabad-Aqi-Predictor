# Islamabad Air Quality Predictor

**3 Day AQI Forecasting System Using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B.svg)](https://streamlit.io/)
[![Hopsworks](https://img.shields.io/badge/MLOps-Hopsworks-00C7B7.svg)](https://www.hopsworks.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A machine learning system that predicts air quality in Islamabad, Pakistan for the next 3 days, enabling citizens to make informed health decisions.

---

## Overview

Islamabad frequently experiences hazardous air quality, particularly during winter months when PM2.5 levels exceed WHO guidelines. This project addresses the critical gap in advance air quality forecasting by providing accurate 72-hour predictions.

### The Problem
- PM2.5 levels regularly exceed 50 µg/m³ (WHO safe limit: 15 µg/m³)
- Citizens lack information to make proactive health decisions

### Our Solution
- Automated ML pipeline predicting PM2.5 levels 72 hours in advance
- Converts predictions to easy-to-understand AQI categories
- Public dashboard accessible to everyone
- Zero manual intervention required

---

## Features

### Machine Learning
- **Three ensemble models**: XGBoost, LightGBM, CatBoost
- **2.30 µg/m³ mean absolute error**
- Automated model selection with overfitting detection
- Version-controlled models in Hopsworks Model Registry

### Automation
- **Hourly data pipeline**: Fetches latest air quality data every hour
- **Daily training pipeline**: Retrains models with new data
- **Automated deployment**: Predictions pushed to dashboard daily
- **GitHub Actions CI/CD**: Zero-downtime updates

### Dashboard Features
- Real-time 3-day AQI forecast
- 72-hour trend visualization
- Color-coded health categories
- Personalized health advisories
- Mobile-responsive design

### MLOps Best Practices
- Feature store management (Hopsworks)
- Model versioning and registry
- Reproducible pipelines
- Automated testing and validation
- Complete audit trail

---

## System Architecture

```
┌─────────────────┐
│  Open-Meteo API │  ← Hourly air quality data
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         GitHub Actions (Automation)          │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │ Hourly:      │      │ Daily (01:00):  │ │
│  │ Fetch data   │      │ Train models    │ │
│  │ Insert to    │      │ Register best   │ │
│  │ Hopsworks    │      │ Generate preds  │ │
│  └──────────────┘      └─────────────────┘ │
└─────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐    ┌──────────────────┐
│ Hopsworks        │    │ GitHub Repo      │
│ • Feature Store  │    │ predictions/     │
│ • Model Registry │    │ (CSV + JSON)     │
└──────────────────┘    └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Streamlit Cloud  │
                        │ (Dashboard)      │
                        └──────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.11+
- Hopsworks account (free tier)
- Git

### Clone Repository
```bash
git clone https://github.com/Ayema-Haris/Islamabad-Aqi-Predictor.git
cd Islamabad-Aqi-Predictor
```

### Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure Environment Variables
Create a `.env` file in the project root:
```env
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT=Islamabad_Aqi_Predictor
```

---

## Usage

### Run Data Pipeline
```bash
python src/feature_store/aqi_feature_pipeline.py
```

### Train Models
```bash
python src/model/train_models.py
```

### Generate Predictions
```bash
python src/forecasting/predict_3_days.py
```

### Launch Dashboard Locally
```bash
streamlit run src/app/streamlit_app.py
```

---

## Project Structure

```
Islamabad-Aqi-Predictor/
│
├── src/
│   ├── feature_store/
│   │   └── aqi_feature_pipeline.py      # Hourly data collection
│   ├── model/
│   │   └── train_models.py              # Model training pipeline
│   ├── forecasting/
│   │   └── predict_3_days.py            # Prediction generation
│   └── app/
│       └── streamlit_app.py             # Dashboard application
│
├── artifacts/
│   ├── predictions/                     # Generated forecasts
│   └── models/                          # Local model backups
│
├── notebooks/
│   └── EDA.ipynb                        # Exploratory data analysis
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml                    # GitHub Actions automation
│
├── requirements.txt                     # Python dependencies
├── requirements-ci.txt                  # CI/CD dependencies
├── .python-version                      # Python version spec
├── .gitignore                          
├── README.md
└── PROJECT_REPORT.docx                  # Detailed documentation
```
---

## Contributions

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/Feature`)
3. Commit your changes (`git commit -m 'Add Feature'`)
4. Push to the branch (`git push origin feature/Feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

