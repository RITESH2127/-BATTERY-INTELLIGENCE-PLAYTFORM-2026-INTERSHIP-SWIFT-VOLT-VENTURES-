$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py --server.address=0.0.0.0 --server.port=$env:APP_PORT
