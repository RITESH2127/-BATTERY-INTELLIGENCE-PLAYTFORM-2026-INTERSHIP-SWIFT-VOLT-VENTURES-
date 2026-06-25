"""
Battery Intelligence Platform — Data Preprocessing Pipeline
=============================================================
Handles data loading, cleaning, validation, scaling, and train-test splitting.
Generates preprocessing reports saved to the reports/ directory.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from src.feature_engineering import engineer_features, SOH_FEATURES, RUL_FEATURES, SOH_TARGET, RUL_TARGET


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load battery cycling data from CSV."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "battery_data.csv"
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run data/generate_dataset.py first."
        )
    df = pd.read_csv(data_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values via forward-fill then backward-fill, then median."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    # Remaining NaNs filled with column median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def detect_outliers(df: pd.DataFrame, columns: list = None, factor: float = 3.0) -> dict:
    """
    Detect outliers using the IQR method.

    Returns
    -------
    dict
        Mapping of column name → number of outliers detected.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c not in ["cycle", "soh", "rul"]]

    outlier_report = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_report[col] = n_outliers
    return outlier_report


def clip_outliers(df: pd.DataFrame, columns: list = None, factor: float = 3.0) -> pd.DataFrame:
    """Clip outliers to the IQR boundary."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c not in ["cycle", "soh", "rul", "battery_id"]]

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = df[col].clip(lower, upper)
    return df


def compute_correlation(df: pd.DataFrame, features: list, target: str) -> pd.Series:
    """Compute Pearson correlation of features with the target variable."""
    corr = df[features + [target]].corr()[target].drop(target)
    return corr.sort_values(ascending=False)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Standardize features using StandardScaler fitted on training data.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Execute the full preprocessing pipeline.

    Returns
    -------
    dict with keys:
        - df_raw: raw loaded DataFrame
        - df_engineered: DataFrame with engineered features
        - soh_data: dict with X_train, X_test, y_train, y_test, scaler
        - rul_data: dict with X_train, X_test, y_train, y_test, scaler
        - report: preprocessing report dict
    """
    print("🔄 Starting preprocessing pipeline...")

    # Step 1: Load
    df_raw = load_data()
    print(f"  ✓ Loaded {len(df_raw)} rows")

    # Step 2: Handle missing values
    df_clean = handle_missing_values(df_raw)
    missing_before = int(df_raw.isna().sum().sum())
    missing_after = int(df_clean.isna().sum().sum())
    print(f"  ✓ Missing values: {missing_before} → {missing_after}")

    # Step 3: Detect & clip outliers
    outlier_report = detect_outliers(df_clean)
    df_clean = clip_outliers(df_clean)
    total_outliers = sum(outlier_report.values())
    print(f"  ✓ Outliers detected & clipped: {total_outliers}")

    # Step 4: Feature engineering
    df_engineered = engineer_features(df_clean)
    print(f"  ✓ Engineered {len(df_engineered.columns) - len(df_raw.columns)} new features")

    # Step 5: Correlation analysis
    soh_corr = compute_correlation(df_engineered, SOH_FEATURES, SOH_TARGET)
    rul_corr = compute_correlation(df_engineered, RUL_FEATURES, RUL_TARGET)

    # Step 6: Train-test split
    # Stratify by battery_id to ensure all batteries in both sets
    X_soh = df_engineered[SOH_FEATURES]
    y_soh = df_engineered[SOH_TARGET]
    X_rul = df_engineered[RUL_FEATURES]
    y_rul = df_engineered[RUL_TARGET]

    X_soh_train, X_soh_test, y_soh_train, y_soh_test = train_test_split(
        X_soh, y_soh, test_size=test_size, random_state=random_state
    )
    X_rul_train, X_rul_test, y_rul_train, y_rul_test = train_test_split(
        X_rul, y_rul, test_size=test_size, random_state=random_state
    )
    print(f"  ✓ Train/test split: {len(X_soh_train)}/{len(X_soh_test)}")

    # Step 7: Feature scaling
    X_soh_train_s, X_soh_test_s, soh_scaler = scale_features(X_soh_train, X_soh_test)
    X_rul_train_s, X_rul_test_s, rul_scaler = scale_features(X_rul_train, X_rul_test)
    print("  ✓ Features scaled (StandardScaler)")

    # Step 8: Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_rows": len(df_raw),
        "total_features_raw": len(df_raw.columns),
        "total_features_engineered": len(df_engineered.columns),
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "outliers_detected": outlier_report,
        "total_outliers": total_outliers,
        "train_size": len(X_soh_train),
        "test_size": len(X_soh_test),
        "soh_feature_correlations": soh_corr.to_dict(),
        "rul_feature_correlations": rul_corr.to_dict(),
        "batteries": df_raw["battery_id"].unique().tolist(),
    }

    # Save report
    report_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
    )
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✓ Report saved: {report_path}")

    print("✅ Preprocessing pipeline complete!\n")

    return {
        "df_raw": df_raw,
        "df_engineered": df_engineered,
        "soh_data": {
            "X_train": X_soh_train_s,
            "X_test": X_soh_test_s,
            "y_train": y_soh_train,
            "y_test": y_soh_test,
            "scaler": soh_scaler,
            "features": SOH_FEATURES,
        },
        "rul_data": {
            "X_train": X_rul_train_s,
            "X_test": X_rul_test_s,
            "y_train": y_rul_train,
            "y_test": y_rul_test,
            "scaler": rul_scaler,
            "features": RUL_FEATURES,
        },
        "report": report,
    }


if __name__ == "__main__":
    result = preprocess_pipeline()
    print(f"SoH features: {len(result['soh_data']['features'])}")
    print(f"RUL features: {len(result['rul_data']['features'])}")
