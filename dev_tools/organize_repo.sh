#!/usr/bin/env bash
set -e
# Create packages
mkdir -p src dashboard assets dev_tools
touch src/__init__.py dashboard/__init__.py

# Move core library files into src/
git mv feature_engineering.py src/feature_engineering.py || mv feature_engineering.py src/ || true
git mv data_preprocessing.py src/data_preprocessing.py || mv data_preprocessing.py src/ || true
git mv model_training.py src/model_training.py || mv model_training.py src/ || true
git mv model_evaluation.py src/model_evaluation.py || mv model_evaluation.py src/ || true
git mv explainability.py src/explainability.py || mv explainability.py src/ || true
git mv utils.py src/utils.py || mv utils.py src/ || true
git mv analytics.py src/analytics.py || mv analytics.py src/ || true

# Move dashboard pages into dashboard/
git mv home.py dashboard/home.py || mv home.py dashboard/ || true
git mv soh_prediction.py dashboard/soh_prediction.py || mv soh_prediction.py dashboard/ || true
git mv rul_prediction.py dashboard/rul_prediction.py || mv rul_prediction.py dashboard/ || true
git mv explainability_dashboard.py dashboard/explainability_dashboard.py || mv explainability_dashboard.py dashboard/ || true
git mv fleet_monitoring.py dashboard/fleet_monitoring.py || mv fleet_monitoring.py dashboard/ || true

# Move assets
mkdir -p assets
git mv style.css assets/style.css || mv style.css assets/ || true

# Optional: remove compiled files (requires confirmation)
# find . -type f -name "*.pyc" -print -delete || true
# find . -type d -name "__pycache__" -print -exec rm -rf {} + || true

echo "Repository reorganized. If you used git mv, commit changes now:"
if git status --porcelain | grep .; then
  git add -A
  git commit -m "chore: reorganize repo into src/ and dashboard/ packages"
  echo "Committed reorganization."
else
  echo "No files were moved (they may already be in place)."
fi

echo "Next steps: run 'pip install -r requirements.txt' and then 'streamlit run app.py'"
