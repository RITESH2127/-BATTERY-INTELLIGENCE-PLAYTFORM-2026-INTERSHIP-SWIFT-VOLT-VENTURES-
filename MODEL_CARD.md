# Model Card

## Project

Battery Intelligence Platform

## Intended use

This project is an engineering and academic prototype for demonstrating machine-learning workflows for EV battery State of Health (SoH) and Remaining Useful Life (RUL) estimation.

Intended uses include:

- ML experimentation and learning
- feature-engineering research
- model comparison
- explainability demonstrations
- dashboard and deployment demonstrations
- internship portfolio presentation

The models are not intended to operate as a certified Battery Management System, safety controller, warranty decision engine, or sole maintenance authority.

## Data

The bundled dataset contains 2,748 simulated battery cycles across four battery identifiers:

- B0005
- B0006
- B0007
- B0018

The data is synthetic/simulated and is modeled around NASA PCoE-style battery-aging profiles for experimentation.

## Prediction tasks

### State of Health

Target: `soh`

Selected model: Gradient Boosting Regressor

Stored cross-validation R2: 0.9999

### Remaining Useful Life

Target: `rul`

Selected model: LightGBM Regressor

Stored cross-validation R2: 0.9840

## Feature engineering

The pipeline derives capacity retention, resistance growth, cycle efficiency, degradation rate, temperature stress, battery wear, rolling thermal indicators, voltage drop, and cumulative thermal stress.

## Evaluation

The stored evaluation uses random row-level splitting and 3-fold cross-validation. Because battery cycles are sequential and correlated within a battery, these results can be optimistic.

Before production use, evaluation should be repeated with:

- battery-level grouped cross-validation
- chronological holdouts
- unseen-battery testing
- independent external datasets
- chemistry and environmental diversity
- sensor-noise and missing-data scenarios

## Explainability

SHAP is used for global and local model explanations. SHAP identifies how model inputs contribute to predictions; it does not prove causal relationships.

## Limitations

- Synthetic training data limits real-world generalization.
- Dataset shift across battery chemistry, manufacturers and operating conditions is not fully represented.
- RUL depends on the selected end-of-life definition.
- Rule-based maintenance recommendations are not manufacturer service thresholds.
- Confidence/stability indicators are not calibrated probabilities.

## Risk controls

Recommended production controls include model versioning, data-quality checks, drift monitoring, independent validation, access control, audit logging, artifact provenance, security scanning and human review for safety-relevant decisions.

## Reproducibility

The repository persists model/scaler artifacts and metadata and provides a reproducible local/Docker workflow with automated smoke tests and GitHub Actions CI.
