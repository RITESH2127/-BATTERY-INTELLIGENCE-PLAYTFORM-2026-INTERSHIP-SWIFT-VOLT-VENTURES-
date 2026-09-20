#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
exec streamlit run app.py --server.address=0.0.0.0 --server.port="${APP_PORT:-8501}"
