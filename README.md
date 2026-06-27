# ⚡ Battery Intelligence Platform
A production-quality Machine Learning demo for EV battery health analytics: State‑of‑Health (SoH) prediction and Remaining Useful Life (RUL) forecasting with an interactive Streamlit dashboard.

[![Repo Size](https://img.shields.io/github/repo-size/RITESH2127/-BATTERY-INTELLIGENCE-PLAYTFORM-2026-INTERSHIP-SWIFT-VOLT-VENTURES-?color=blue)]()
[![Last Commit](https://img.shields.io/github/last-commit/RITESH2127/-BATTERY-INTELLIGENCE-PLAYTFORM-2026-INTERSHIP-SWIFT-VOLT-VENTURES-?color=informational)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-%3E%3D1.0-red?logo=streamlit)]()
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20RF-green)]()
[![XAI](https://img.shields.io/badge/XAI-SHAP-orange)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## Table of Contents
- [Project Overview](#project-overview)
- [Stack](#stack)
- [Why this project exists](#why-this-project-exists)
- [Problem Statement](#problem-statement)
- [Solution Summary](#solution-summary)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Architecture & Flow](#architecture--flow)
- [Quick Start — Run the app (recommended)](#quick-start--run-the-app-recommended)
- [Prerequisites](#prerequisites)
- [Step-by-step Setup & Run (manual)](#step-by-step-setup--run-manual)
- [Data & Models](#data--models)
- [Important Implementation Notes & Known Issues](#important-implementation-notes--known-issues)
- [Contributing](#contributing)
- [Credits & Acknowledgements](#credits--acknowledgements)
- [License & Authors](#license--authors)
- [Contact & Support](#contact--support)

---

## Project Overview
Battery Intelligence Platform demonstrates end-to-end ML for EV battery health: synthetic data generation, preprocessing and feature engineering, model training (SoH & RUL), explainability (SHAP), and an interactive Streamlit dashboard for demo and analysis.

### Stack
- **Language(s):** Python 3.10+
- **Framework / runtime:** Streamlit dashboard; scikit-learn / XGBoost / LightGBM for models
- **Notable libraries:** pandas, numpy, scikit-learn, xgboost, lightgbm, joblib, shap, plotly

---

## Why this project exists
EV battery health matters for safety, warranty management, and lifecycle planning. This repo provides a compact, explainable pipeline that shows how SoH and RUL can be estimated from cycling telemetry and domain features.

---

## Problem Statement
Battery degradation is nonlinear and influenced by usage, thermal stress, internal resistance, and operating conditions. This project shows a reproducible approach to build models and visualizations that help engineers inspect and monitor battery health.

---

## Solution Summary
- Synthetic dataset generator that mimics battery aging patterns.
- Feature engineering that captures capacity fade, resistance growth and thermal stress.
- Training pipeline with model selection and persistence (joblib artifacts).
- Streamlit dashboard with pages for Home, SoH, RUL, Analytics, Explainability, and Fleet Monitoring.
- SHAP-based explainability for global and local interpretation.

---

## Key Features
- SoH prediction (0–100%) and RUL prediction (remaining cycles).
- Multiple pre-trained models included for fast demos (.joblib files).
- Automated model selection using cross-validation (R²).
- SHAP explainability: global feature importance and per-prediction explanations.
- Streamlit dashboard with polished styling (custom CSS available).
- Preprocessing report (JSON) describing dataset statistics and data quality.

---

## Repository Structure (top-level files)
```
LICENSE
README.md
__init__.py
__pycache__/            (compiled files; recommend removing from VCS)
analytics.py
app.py
battery_data.csv
config.toml
data_preprocessing.py
explainability.py
explainability_dashboard.py
feature_engineering.py
fleet_monitoring.py
generate_dataset.py
home.py
model_evaluation.py
model_metadata.json
model_training.py
preprocessing_report.json
project_report.md
requirements.txt
rul_prediction.py
rul_*.joblib
soh_*.joblib
style.css
utils.py
```
Note: Many source files are at the repository root; imports in the code expect a package layout (e.g., `src.*` and `dashboard.*`). See "Quick Start" for a short fix.

---

## Architecture & Flow
I fixed and clarified the architecture section: the diagram and the runtime description below accurately reflect how data flows through the repo and what modules are responsible for each step.

Mermaid diagram (logical flow):

```mermaid
flowchart LR
  A[Data generator / battery_data.csv] --> B[Data Preprocessing (data_preprocessing.py)]
  B --> C[Feature Engineering (feature_engineering.py)]
  C --> D[Train/Test Split + Scaling (scaler joblib)]
  D --> E{Model Training (model_training.py)}
  E -->|GridSearchCV| F[Candidate Models: RandomForest, XGBoost, LightGBM]
  F --> G[Persist models & scalers (.joblib) + model_metadata.json]
  G --> H[Streamlit Dashboard (app.py) — loads artifacts]
  H --> I[Pages: Home, SoH Prediction, RUL Prediction, Analytics, Explainability, Fleet Monitoring]
  H --> J[SHAP Explainability (explainability.py / explainability_dashboard.py)]
```

How it fits together (runtime):
- app.py is the Streamlit entrypoint; it routes to page modules (home.py, soh_prediction.py, rul_prediction.py, analytics.py, explainability_dashboard.py, fleet_monitoring.py).
- Data preparation: generate_dataset.py -> data_preprocessing.py -> feature_engineering.py -> preprocessing_report.json.
- Training: model_training.py calls preprocessing, runs GridSearchCV across candidate estimators (RandomForest/XGBoost/LightGBM), saves best models and scalers along with model_metadata.json.
- Dashboard pages load the persisted artifacts for inference and analysis; explainability code uses SHAP to produce local/global explanations.

---

## Quick Start — Run the app (recommended)
The shortest path to run the dashboard locally (from project root):

1) Optional: reorganize files so imports resolve (one-time):
```bash
# Recommended automated reorganization (creates packages so imports like `src.*` and `dashboard.*` work)
mkdir -p src dashboard assets
# Move core modules into src/ and dashboard pages into dashboard/ (manual or via script)
# See README notes or ask me to add dev_tools/organize_repo.sh to automate this.
```

2) Setup environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate         # macOS/Linux
.\.venv\Scripts\activate       # Windows (PowerShell)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the Streamlit app:
```bash
streamlit run app.py
```

4) Open http://localhost:8501 in your browser.

If imports fail, create `src/` and `dashboard/` directories and move the corresponding modules into them, or request that I add an automated reorganization script and commit it for you.

---

## Prerequisites
- Python 3.10+
- Virtual environment recommended (venv/virtualenv)
- OS: Linux / macOS / Windows

Dependencies are listed in requirements.txt (pinned versions recommended when deploying).

---

## Important Implementation Notes & Known Issues
- Import layout: Several modules import as `src.*` / `dashboard.*` but files are at the repo root. Easiest fix: move files into `src/` and `dashboard/` packages (I can add a script to do this).
- Compiled .pyc files are present in the repository; they should be removed and .gitignore updated to exclude them.
- Model artifacts (.joblib) are included for demo; ensure compatibility between the versions used to serialize and your runtime XGBoost/LightGBM.
- Dataset is synthetic — do not use results as real-world production predictions without validating on real telemetry.

---

## Data & Models
- battery_data.csv — sample/synthetic dataset.
- preprocessing_report.json — data quality and correlation info.
- Model artifacts in the root (soh_*.joblib, rul_*.joblib) and model_metadata.json (best model names and CV scores).

---

## Contributing
Contributions are welcome — bug fixes, adding tests, improving packaging/layout, adding CI, and extending explainability.

Suggested workflow: fork, create a branch, add tests, open a PR.

---

## Credits & Acknowledgements
- NASA battery dataset (pattern inspiration)
- SHAP, Streamlit, XGBoost, LightGBM

---

## License & Authors
MIT License — see LICENSE file.
Author: RITESH2127

---

## Contact & Support
Open an issue for questions or problems. Include command output and steps to reproduce.

---

Notes:
- I fixed the Architecture & Flow section to be clear and actionable, removed truncated lines, and improved the Quick Start instructions.
- If you want, I can:
  - add dev_tools/organize_repo.sh and commit it to automatically reorganize files into `src/` and `dashboard/`,
  - remove the compiled .pyc files from the repo and add a .gitignore,
  - or open a pull request that applies these repository-cleanup changes.
