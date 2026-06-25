# ⚡ Battery Intelligence Platform

## EV Battery Health & Remaining Useful Life Prediction

A production-quality Machine Learning application for predicting **State of Health (SoH)** and **Remaining Useful Life (RUL)** of lithium-ion EV batteries. Built with Streamlit, scikit-learn, XGBoost, LightGBM, and SHAP.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![ML](https://img.shields.io/badge/ML-XGBoost%20|%20LightGBM%20|%20RF-green)
![XAI](https://img.shields.io/badge/XAI-SHAP-orange)

---

## 🎯 Features

### Machine Learning
- **SoH Prediction** — Predicts battery health as a percentage (0–100%)
- **RUL Prediction** — Estimates remaining charge cycles before end-of-life
- **6 ML Models** — Random Forest, XGBoost, Gradient Boosting, LightGBM
- **Auto Model Selection** — Automatically selects the best-performing model by R²
- **10+ Engineered Features** — Capacity retention, resistance growth, wear index, etc.

### Explainable AI
- **SHAP Feature Importance** — Global feature ranking by mean |SHAP value|
- **Prediction Breakdown** — Local waterfall explanations for individual predictions
- **Feature Interaction** — Dependence plots showing feature relationships

### Dashboard (6 Pages)
| Page | Description |
|------|-------------|
| 🏠 Home Dashboard | KPI cards, fleet health overview, status distribution, trend chart |
| 🔋 SoH Prediction | Input form → gauge meter, health classification, confidence score |
| 🔄 RUL Prediction | Input form → remaining life gauge, degradation forecast curve |
| 📊 Analytics | Capacity fade, temperature impact, cycle analysis, correlation heatmap |
| 🧠 Explainable AI | SHAP importance, prediction breakdown, top factors analysis |
| 🚗 Fleet Monitoring | 25 simulated EVs with search, filters, sorting, drill-down |

### Design
- 🌑 **Dark Mode** with electric blue (#00D1FF) accents
- 💎 **Glassmorphism** cards with backdrop blur
- ✨ **Micro-animations** (shimmer, gradient rotation, hover effects)
- 📱 **Responsive** layout

---

## 🏗️ Architecture

```
battery-intelligence-platform/
├── .streamlit/config.toml          # Dark theme configuration
├── data/
│   ├── generate_dataset.py         # Synthetic NASA-style data generator
│   └── battery_data.csv            # Generated dataset
├── src/
│   ├── data_preprocessing.py       # Cleaning, scaling, train-test split
│   ├── feature_engineering.py      # 10+ derived features
│   ├── model_training.py           # Train 6 models + GridSearchCV
│   ├── model_evaluation.py         # MAE, MSE, RMSE, R² + Plotly charts
│   ├── explainability.py           # SHAP TreeExplainer
│   └── utils.py                    # Risk classification, recommendations
├── dashboard/
│   ├── home.py                     # Home Dashboard
│   ├── soh_prediction.py           # SoH Prediction page
│   ├── rul_prediction.py           # RUL Prediction page
│   ├── analytics.py                # Analytics Dashboard
│   ├── explainability_dashboard.py # Explainable AI page
│   └── fleet_monitoring.py         # Fleet Monitoring page
├── models/                         # Saved .joblib models
├── assets/style.css                # Premium CSS theme
├── reports/                        # Preprocessing reports
├── app.py                          # Main Streamlit entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
python -c "from data.generate_dataset import generate_battery_data; generate_battery_data()"
python -c "from src.model_training import train_all_models; train_all_models()"
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Model Performance

| Task | Model | R² Score |
|------|-------|----------|
| SoH | Random Forest | ~0.98+ |
| SoH | XGBoost | ~0.99+ |
| SoH | Gradient Boosting | ~0.98+ |
| RUL | Random Forest | ~0.95+ |
| RUL | XGBoost | ~0.96+ |
| RUL | LightGBM | ~0.96+ |

*Scores are on synthetic data modeled on NASA PCoE battery aging patterns.*

---

## 🔬 Data

The platform uses a **synthetic dataset** modeled on the [NASA Battery Aging Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#battery). It simulates 4 batteries (B0005, B0006, B0007, B0018) with:

- Realistic non-linear capacity fade curves
- Knee-point degradation acceleration
- Temperature-dependent aging
- Internal resistance growth
- ~2,750 total cycle records

The generator (`data/generate_dataset.py`) can be replaced with real NASA `.mat` data loading.

---

## 🧪 Engineered Features

| Feature | Description |
|---------|-------------|
| Capacity Retention Rate | Current capacity / initial capacity |
| Resistance Growth Rate | Resistance change from baseline |
| Cycle Efficiency | Discharge power / charge power ratio |
| Degradation Rate | Rolling capacity loss per cycle |
| Temperature Stress Score | Deviation from optimal range (20–30°C) |
| Battery Wear Index | Composite of resistance + capacity fade |
| Avg Charge Temperature | Rolling mean of charging temperature |
| Voltage Drop | Charge voltage − measured voltage |
| Cumulative Temp Stress | Accumulated temperature stress over cycles |

---

## 🎯 Target Users

- EV Manufacturers
- Battery Manufacturers
- EV Fleet Operators
- EV Service Centers
- Research Organizations
- Energy Storage Companies

---

## 📝 License

This project is for educational and demonstration purposes.

---

## 🙏 Acknowledgments

- [NASA PCoE](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/) — Battery aging dataset reference
- [SHAP](https://github.com/shap/shap) — Explainable AI
- [Streamlit](https://streamlit.io) — Dashboard framework
- [XGBoost](https://xgboost.readthedocs.io) / [LightGBM](https://lightgbm.readthedocs.io) — ML models
