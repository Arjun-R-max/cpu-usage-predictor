# app.py
import streamlit as st
import os
import requests
import mlflow
import joblib
from dagshub.streaming import install_hooks
install_hooks()

# config
DAGSHUB_USER = os.getenv('DAGSHUB_USER', 'Arjun-R-max')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', 'cpu-usage-predictor')
MLFLOW_URI = f'https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow'

st.title("CPU Usage Model Dashboard")

st.markdown("## Experiments from DagsHub MLflow")
# List experiments via MLflow REST API (public endpoints)
# For simple demo, fetch experiments list
exp_list_url = f'{MLFLOW_URI}/api/2.0/mlflow/experiments/list'
resp = requests.get(exp_list_url)
if resp.status_code == 200:
    data = resp.json()
    st.write(data)  # for simplicity; you can parse and show table
else:
    st.write("Couldn't fetch experiments (check DAGSHUB credentials or repo visibility). Status:", resp.status_code)

st.markdown("## Make a prediction")

# inputs
cpu_request = st.number_input('cpu_request', value=0.5)
mem_request = st.number_input('mem_request', value=256.0)
cpu_limit = st.number_input('cpu_limit', value=1.0)
mem_limit = st.number_input('mem_limit', value=512.0)
runtime_minutes = st.number_input('runtime_minutes', value=60)
controller_kind = st.selectbox('controller_kind', options=['deployment','statefulset','daemonset'])  # adjust

# load model via DDA: path in repo (DVC-tracked path or models/ path)
model_path = 'model.joblib'   # if you DVC-added it at repo root, or adjust path in repo
try:
    model = joblib.load(model_path)
    st.success("Loaded model from DagsHub (streamed).")
except Exception as e:
    st.error(f"Couldn't load model: {e}")
    model = None

if st.button('Predict'):
    if model is None:
        st.error("No model loaded")
    else:
        X = [[cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_kind]]
        pred = model.predict(X)[0]
        st.write("Predicted CPU usage:", float(pred))
