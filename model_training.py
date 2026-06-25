"""
Battery Intelligence Platform — Model Training
=================================================
Trains SoH and RUL prediction models, performs hyperparameter tuning,
selects the best model, and saves all artifacts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime

from src.data_preprocessing import preprocess_pipeline


# ---------- Model Configurations ---------- #

SOH_MODELS = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1],
        },
    },
}

RUL_MODELS = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.05, 0.1],
        },
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
            "learning_rate": [0.05, 0.1],
        },
    },
}


def _train_models(model_configs: dict, X_train, y_train, task_name: str) -> dict:
    """
    Train multiple models with GridSearchCV and return results.

    Returns
    -------
    dict
        Mapping of model_name → {"model": fitted_model, "best_params": dict, "cv_score": float}
    """
    results = {}
    for name, config in model_configs.items():
        print(f"  Training {task_name} — {name}...")
        grid = GridSearchCV(
            config["model"],
            config["params"],
            cv=3,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        results[name] = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "cv_score": round(grid.best_score_, 4),
        }
        print(f"    ✓ {name}: CV R² = {grid.best_score_:.4f} | Params: {grid.best_params_}")
    return results


def _select_best_model(results: dict) -> tuple:
    """Select the best model by CV R² score."""
    best_name = max(results, key=lambda k: results[k]["cv_score"])
    return best_name, results[best_name]


def train_all_models() -> dict:
    """
    Execute full training pipeline: preprocess → train → select → save.

    Returns
    -------
    dict with keys:
        - soh_models: dict of all SoH model results
        - rul_models: dict of all RUL model results
        - soh_best: (name, model_info)
        - rul_best: (name, model_info)
        - soh_scaler, rul_scaler: fitted scalers
        - soh_features, rul_features: feature lists
        - preprocessing_result: full preprocessing output
    """
    # Step 1: Preprocess
    prep = preprocess_pipeline()

    soh_data = prep["soh_data"]
    rul_data = prep["rul_data"]

    # Step 2: Train SoH models
    print("\n🔄 Training SoH Prediction Models...")
    soh_results = _train_models(
        SOH_MODELS, soh_data["X_train"], soh_data["y_train"], "SoH"
    )
    soh_best_name, soh_best_info = _select_best_model(soh_results)
    print(f"\n  🏆 Best SoH Model: {soh_best_name} (R² = {soh_best_info['cv_score']:.4f})")

    # Step 3: Train RUL models
    print("\n🔄 Training RUL Prediction Models...")
    rul_results = _train_models(
        RUL_MODELS, rul_data["X_train"], rul_data["y_train"], "RUL"
    )
    rul_best_name, rul_best_info = _select_best_model(rul_results)
    print(f"\n  🏆 Best RUL Model: {rul_best_name} (R² = {rul_best_info['cv_score']:.4f})")

    # Step 4: Save models
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    os.makedirs(models_dir, exist_ok=True)

    # Save all SoH models
    for name, info in soh_results.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(info["model"], os.path.join(models_dir, f"soh_{safe_name}.joblib"))

    # Save all RUL models
    for name, info in rul_results.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(info["model"], os.path.join(models_dir, f"rul_{safe_name}.joblib"))

    # Save scalers
    joblib.dump(soh_data["scaler"], os.path.join(models_dir, "soh_scaler.joblib"))
    joblib.dump(rul_data["scaler"], os.path.join(models_dir, "rul_scaler.joblib"))

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "soh_best_model": soh_best_name,
        "soh_best_cv_score": soh_best_info["cv_score"],
        "soh_best_params": soh_best_info["best_params"],
        "rul_best_model": rul_best_name,
        "rul_best_cv_score": rul_best_info["cv_score"],
        "rul_best_params": rul_best_info["best_params"],
        "soh_features": soh_data["features"],
        "rul_features": rul_data["features"],
        "all_soh_scores": {k: v["cv_score"] for k, v in soh_results.items()},
        "all_rul_scores": {k: v["cv_score"] for k, v in rul_results.items()},
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\n✅ All models saved to {models_dir}")

    return {
        "soh_models": soh_results,
        "rul_models": rul_results,
        "soh_best": (soh_best_name, soh_best_info),
        "rul_best": (rul_best_name, rul_best_info),
        "soh_scaler": soh_data["scaler"],
        "rul_scaler": rul_data["scaler"],
        "soh_features": soh_data["features"],
        "rul_features": rul_data["features"],
        "preprocessing_result": prep,
    }


if __name__ == "__main__":
    train_all_models()
