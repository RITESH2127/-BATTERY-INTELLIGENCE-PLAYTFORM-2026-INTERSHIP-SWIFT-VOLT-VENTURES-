# ⚡ Battery Intelligence Platform
A production-quality Machine Learning project for EV battery health analytics: State‑of‑Health (SoH) prediction and Remaining Useful Life (RUL) forecasting, bundled with a premium Streamlit dashboard and Explainable AI (SHAP) visualizations.

[![Repo Size](https://img.shields.io/github/repo-size/RITESH2127/-BATTERY-INTELLIGENCE-PLAYTFORM-2026-INTERSHIP-SWIFT-VOLT-VENTURES-?color=blue)]()
[![Last Commit](https://img.shields.io/github/last-commit/RITESH2127/-BATTERY-INTELLIGENCE-PLAYTFORM-2026-INTERSHIP-SWIFT-VOLT-VENTURES-?color=informational)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-%3E%3D1.0-red?logo=streamlit)]()
[![ML](https://img.shields.io/badge/ML-XGBoost%20|%20LightGBM%20|%20RF-green)]()
[![XAI](https://img.shields.io/badge/XAI-SHAP-orange)]()
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey)]()

---

<!-- TOC -->
## Table of Contents
- [Project Overview](#project-overview)
- [Why this project exists](#why-this-project-exists)
- [Problem Statement](#problem-statement)
- [Solution Summary](#solution-summary)
- [Key Features](#key-features)
- [Feature Highlights](#feature-highlights)
- [Repository Structure (actual)](#repository-structure-actual)
- [Architecture & Flow](#architecture--flow)
- [Quick Start — Run the app (recommended)](#quick-start---run-the-app-recommended)
- [Prerequisites](#prerequisites)
- [Automated Repair / Run Fix](#automated-repair--run-fix)
- [Step-by-step Setup & Run (manual)](#step-by-step-setup--run-manual)
- [Data & Models](#data--models)
- [Important Implementation Notes & Known Issues](#important-implementation-notes--known-issues)
- [APIs & Scripts](#apis--scripts)
- [Testing & Evaluation](#testing--evaluation)
- [Security & Privacy Considerations](#security--privacy-considerations)
- [Roadmap & Future Work](#roadmap--future-work)
- [Contributing](#contributing)
- [Credits & Acknowledgements](#credits--acknowledgements)
- [License & Authors](#license--authors)
- [Contact & Support](#contact--support)
- [Star the repo](#star-the-repo)
<!-- /TOC -->

---

## Project Overview
Battery Intelligence Platform is a focused ML application demonstrating how to predict EV battery State‑of‑Health (SoH) and Remaining Useful Life (RUL) using engineered features, ensemble models (Random Forest, XGBoost, LightGBM, Gradient Boosting), and provide explainability with SHAP. The project includes a Streamlit dashboard (multi-page UI) for interactive prediction, fleet monitoring, analytics, and XAI exploration.

Stack (observed from code)
- Language(s): Python 3.10+
- Runtime / Framework: Streamlit (dashboard), scikit-learn / XGBoost / LightGBM (models)
- Notable libraries: pandas, numpy, plotly, scikit-learn, xgboost, lightgbm, joblib, shap

---

## Why this project exists
EV battery health is critical for safety, warranty management, and lifecycle planning. This repository showcases a compact, end‑to‑end pipeline for:
- generating or ingesting battery cycling data,
- cleaning and engineering domain features,
- training and evaluating ensemble models for SoH and RUL,
- surfacing per‑prediction and global explainability,
- delivering the results through an interactive, production‑style dashboard.

---

## Problem Statement
Battery degradation is nonlinear and influenced by usage (cycles), thermal stress, internal resistance, and operating conditions. Predicting SoH and RUL requires robust features, careful preprocessing, model selection, and interpretability for real-world adoption.

---

## Solution Summary
- Synthetic dataset generator that mimics NASA battery aging patterns (cycle profiles, temperature effects).
- Feature engineering (capacity retention, resistance growth, degradation rate, temperature stress, wear index, cumulative stress, etc.).
- Training pipeline that runs GridSearchCV across candidate models and persists artifacts (joblib models + scalers + metadata).
- Streamlit dashboard with pages for Home, SoH prediction, RUL prediction, Analytics, Explainability (SHAP), and Fleet Monitoring.
- Pre-built model artifacts are included (joblib files) so you can run predictions/demo quickly.

---

## Key Features
- SoH prediction (0–100%) with model confidence estimate.
- RUL prediction (remaining cycles) using ensembles.
- Multiple pre-trained model artifacts available (.joblib files).
- Automated model selection based on CV R².
- SHAP‑based explainability: global importance, local waterfall explanations.
- Streamlit dashboard with premium styling (dark theme, glassmorphism CSS).
- Preprocessing report output (JSON) describing dataset, correlations, and outliers.

---

## Feature Highlights
- 10+ engineered, domain-specific features to improve model generalization:
  - capacity_retention_rate, resistance_growth_rate, cycle_efficiency, degradation_rate, temperature_stress_score, battery_wear_index, avg_charge_temperature, avg_discharge_temperature, voltage_drop, cumulative_temp_stress.
- Model training search over a compact but effective hyperparameter grid for each algorithm.
- Rich Plotly visualizations set to a dark UI theme and consistent color palette.

---

## Repository Structure (actual files observed)
This is the repository content as present at the top level (files are shown exactly as in the project root):

```
LICENSE
README.md
__init__.py
__init__.cpython-312.pyc
__pycache__/
analytics.py
analytics.cpython-312.pyc
app.py
battery_data.csv
config.toml
data_preprocessing.py
data_preprocessing.cpython-312.pyc
explainability.py
explainability.cpython-312.pyc
explainability_dashboard.py
explainability_dashboard.cpython-312.pyc
feature_engineering.py
feature_engineering.cpython-312.pyc
fleet_monitoring.py
fleet_monitoring.cpython-312.pyc
generate_dataset.py
generate_dataset.cpython-312.pyc
home.py
home.cpython-312.pyc
model_evaluation.py
model_evaluation.cpython-312.pyc
model_metadata.json
model_training.py
model_training.cpython-312.pyc
preprocessing_report.json
project_report.md
requirements.txt
rul_prediction.py
rul_prediction.cpython-312.pyc
rul_lightgbm.joblib
rul_random_forest.joblib
rul_scaler.joblib
rul_xgboost.joblib
soh_prediction.py
soh_prediction.cpython-312.pyc
soh_gradient_boosting.joblib
soh_random_forest.joblib
soh_scaler.joblib
soh_xgboost.joblib
style.css
utils.py
utils.cpython-312.pyc
```

Note: Several compiled .pyc files are included; the source .py files are the primary implementation.

---

## Architecture & Flow

Mermaid diagram (logical flow):

```mermaid
flowchart LR
  A[Data generator / battery_data.csv] --> B[Data Preprocessing]
  B --> C[Feature Engineering]
  C --> D[Train/Test Split + Scaling]
  D --> E{Model Training}
  E -->|GridSearchCV| F[RandomForest / XGBoost / LightGBM / GB]
  F --> G[Persist models (.joblib) + scaler + metadata]
  G --> H[Streamlit Dashboard (app.py)]
  H --> I[Pages: Home, SoH, RUL, Analytics, Explainability, Fleet]
  H --> J[SHAP Explainability]
```

How it fits together (runtime shape)
- The main application entry point is app.py — it presents a Streamlit sidebar and routes to page modules (SoH, RUL, Analytics, Explainability, Fleet).
- Preprocessing pipeline (data_preprocessing.py) loads data, handles missing values, detects/clips outliers, runs feature engineering, computes correlations, performs train/test splits and scales features.
- Model training (model_training.py) calls preprocessing, trains models using GridSearchCV, saves models/scalers, and writes model_metadata.json for the dashboard to report status.
- Dashboard pages (home.py, soh_prediction.py, rul_prediction.py, analytics.py, explainability_dashboard.py, fleet_monitoring.py) consume saved model artifacts and preprocessing metadata to render visualizations and predictions.

---

## Quick Start — Run the app (recommended)
The repository contains pre-trained model artifacts (joblib) and a model metadata file so you can try the dashboard locally.

Important: There is a small mismatch between how modules are imported in the code (some imports reference `src.*` and `dashboard.*`) and how files are placed at the repository root. See the "Important Implementation Notes & Known Issues" section below for automatic repair commands. If you want the fastest path to demo the UI, use the automated repair script (recommended) below, which moves files into the expected package structure.

Automated repair (one-line, from project root):
```bash
# Option A: automated reorganization (recommended)
bash ./dev_tools/organize_repo.sh
# then:
pip install -r requirements.txt
streamlit run app.py
```

If you prefer manual steps, follow the "Step-by-step Setup & Run (manual)" section.

---

## Prerequisites
- Python 3.10+ (tested with 3.10/3.11)
- Recommended: virtualenv / venv
- OS: Linux / macOS / Windows (Streamlit + Python compatible)

Observed runtime dependencies (based on imports in source):
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- joblib
- plotly
- shap

Install (example):
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.\.venv\Scripts\activate         # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Automated Repair / Run Fix
Because the repository currently contains top-level modules while some imports expect packages (`src.*`, `dashboard.*`), the repository includes a recommended reorganization to match imports and enable the Streamlit app to run without modifying code.

Create a small script (dev_tools/organize_repo.sh) — recommended contents:

```bash
#!/usr/bin/env bash
set -e
# Create packages
mkdir -p src dashboard
touch src/__init__.py dashboard/__init__.py

# Move core library files into src/
git mv feature_engineering.py src/feature_engineering.py || mv feature_engineering.py src/
git mv data_preprocessing.py src/data_preprocessing.py || mv data_preprocessing.py src/
git mv model_training.py src/model_training.py || mv model_training.py src/
git mv model_evaluation.py src/model_evaluation.py || mv model_evaluation.py src/
git mv explainability.py src/explainability.py || mv explainability.py src/
git mv utils.py src/utils.py || mv utils.py src/
git mv analytics.py src/analytics.py || mv analytics.py src/

# Move dashboard pages into dashboard/
git mv home.py dashboard/home.py || mv home.py dashboard/
git mv soh_prediction.py dashboard/soh_prediction.py || mv soh_prediction.py dashboard/
git mv rul_prediction.py dashboard/rul_prediction.py || mv rul_prediction.py dashboard/
git mv explainability_dashboard.py dashboard/explainability_dashboard.py || mv explainability_dashboard.py dashboard/
git mv fleet_monitoring.py dashboard/fleet_monitoring.py || mv fleet_monitoring.py dashboard/

# Move assets
mkdir -p assets
git mv style.css assets/style.css || mv style.css assets/

echo "Repository reorganized. Please run 'pip install -r requirements.txt' and then 'streamlit run app.py'"
```

Save as dev_tools/organize_repo.sh, make executable (chmod +x), run it once, then start the app:
```bash
bash dev_tools/organize_repo.sh
pip install -r requirements.txt
streamlit run app.py
```

This places code where imports expect them: app.py imports dashboard.*, and many modules import src.*.

---

## Step-by-step Setup & Run (manual)
1. Clone the repository:
   ```bash
   git clone https://github.com/RITESH2127/-BATTERY-INTELLIGENCE-PLAYTFORM-2026-INTERSHIP-SWIFT-VOLT-VENTURES-.git
   cd -BATTERY-INTELLIGENCE-PLAYTFORM-2026-INTERSHIP-SWIFT-VOLT-VENTURES-
   ```

2. (Recommended) Apply the automated reorganization above or manually create the package layout:
   - Create `src/` and `dashboard/` directories and add `__init__.py`.
   - Move core modules into `src/` and dashboard pages into `dashboard/`.
   - Move `style.css` to `assets/style.css`.

   This step aligns runtime imports with file locations.

3. Create & activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. (Optional) Generate synthetic dataset (if you want to re-run training):
   ```bash
   python -c "from src.generate_dataset import generate_battery_data; generate_battery_data()"
   # or
   python src/generate_dataset.py
   ```

5. (Optional) Run preprocessing & training:
   ```bash
   python -c "from src.model_training import train_all_models; train_all_models()"
   # or
   python src/model_training.py
   ```

   Trained models and scalers will be saved in `models/` and metadata in `models/model_metadata.json`.

6. Launch the dashboard:
   ```bash
   streamlit run app.py
   # If import errors remain: ensure current directory contains `src/` and `dashboard/` and that __init__.py files exist
   ```

7. Open http://localhost:8501 in your browser.

---

## Data & Models
- battery_data.csv — synthetic dataset included (sample cycling records).
- Preprocessing report: preprocessing_report.json (contains correlations, train/test sizes, outlier counts).
- Model artifacts included (examples):
  - soh_random_forest.joblib
  - soh_xgboost.joblib
  - soh_gradient_boosting.joblib
  - rul_random_forest.joblib
  - rul_xgboost.joblib
  - rul_lightgbm.joblib
  - soh_scaler.joblib
  - rul_scaler.joblib
  - model_metadata.json (contains best model names and CV scores)

The SoH and RUL joblib files are ready for inference from the Streamlit UI.

---

## Important Implementation Notes & Known Issues (must read)
- Module import mismatch: Many source files import modules as `src.*` or `dashboard.*`, but the repository's top-level layout places those .py files at the root. The Streamlit entrypoint (app.py) expects a Python package `dashboard` to exist. Because of this, the app may raise ImportError until files are reorganized (see Automated Repair above).
- Compiled .pyc files are present in the repo; they are not necessary and can be removed (`git rm`) to keep source clean.
- The dataset is synthetic and used for demonstration and evaluation — numbers and model metrics reflect this synthetic dataset and should not be treated as production‑grade predictions without validation on real battery data.
- Model artifacts are stored as joblib files; ensure versions of XGBoost/LightGBM match what was used to serialize — upgrading/downgrading can cause compatibility issues.

---

## APIs & Scripts (what each major script does)
- app.py — Streamlit entrypoint and multi-page router.
- generate_dataset.py — synthetic battery dataset generator (NASA-like patterns).
- data_preprocessing.py — loading, missing-value imputation, outlier detection/clipping, feature engineering hook call, train/test split, scaling, and report generation.
- feature_engineering.py — domain feature construction used for SoH and RUL models.
- model_training.py — orchestrates training of candidate models, hyperparameter search, selects best by CV R², saves models/scalers and model_metadata.json.
- model_evaluation.py — utilities to compute MAE/MSE/RMSE/R² and Plotly visualizations (dark theme).
- explainability.py & explainability_dashboard.py — SHAP explainers for global/local interpretation and the dashboard page.
- soh_prediction.py / rul_prediction.py — Streamlit pages for interactive prediction (collects inputs, creates engineered features from inputs, scales, and calls model.predict).
- analytics.py — analytics dashboard (heatmaps, capacity fade charts).
- fleet_monitoring.py — simulated fleet UI & sample filters.
- utils.py — helper functions: risk classification, icons, color maps, recommendation logic, synthetic fleet generator.

---

## Testing & Evaluation
- Model training uses GridSearchCV with cv=3 and scoring="r2".
- Preprocessing writes a preprocessing_report.json which includes feature correlations and data quality metrics.
- Included model_metadata.json contains CV R² scores per algorithm — use these to verify chosen models.
- For reproducible evaluation, set random_state parameters (present in training code).

---

## Security & Privacy Considerations
- This repository uses synthetic sample data. If you substitute real battery telemetry:
  - Remove or secure any PII (vehicle IDs, owner info).
  - Ensure secure handling and storage of model artifacts and telemetry.
  - Consider role-based access for dashboards and protect endpoints behind authentication for production deployments.

---

## Roadmap & Future Improvements
- Ingest real-world NASA `.mat` datasets and extend the data loader.
- Add unit tests (pytest) for preprocessing, feature engineering and model outputs.
- CI workflow for automated training and model validation.
- Dockerfile and docker-compose for containerized deployment.
- Add authentication and exportable reports (PDF/CSV).
- Convert dashboard into a deployable web service (FastAPI backend + React/Streamlit frontend separation) for scale.

---

## Contributing
Thank you for considering contributing! Typical contributions:
- Bug fixes (particularly the import layout / packaging)
- New explainability visualizations
- Additional models or performance improvements
- Tests and CI integration

Suggested workflow:
1. Fork the repo
2. Create a feature branch
3. Add tests where appropriate
4. Open a PR with a clear description and reproducible steps

Please follow idiomatic Python style (PEP8) and include unit tests for new logic.

---

## Credits & Acknowledgements
- NASA PCoE battery dataset (pattern inspiration)
- SHAP — explainability
- Streamlit — interactive dashboard framework
- XGBoost & LightGBM — performant gradient-boosted models

---

## License & Authors
- See the LICENSE file in the repository root for license terms.
- Author: RITESH2127 (repository owner) — see GitHub profile for contact details.

---

## Contact & Support
For questions, issues, or feature requests:
- Open an issue in this repository.
- Provide console output and steps to reproduce when filing issues.

---

## Star the repo
If you find this project useful, please give it a star ⭐ to help others discover it.

---

> Closing note  
> This README describes the repository as currently implemented and includes recommended quick fixes to run the Streamlit dashboard locally. If you want, I can:
> - produce the dev_tools/organize_repo.sh script and add it to the repo, or
> - produce a minimal setup patch (git patch) that reorganizes files into `src/` and `dashboard/` and adds __init__.py files so the app runs without changes.
