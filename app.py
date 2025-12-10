# app.py
import streamlit as st
import os
import requests
import mlflow
import joblib
import pandas as pd
from dagshub.streaming import install_hooks

install_hooks(
    repo_url=f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.git"
)

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
DAGSHUB_USER = os.getenv('DAGSHUB_USER', 'Arjun-R-max')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', 'cpu-usage-predictor')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')  # Required if repo is private

MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("⚡ CPU Usage Prediction Dashboard")

# --------------------------------------------------------------------
# 1. SHOW EXPERIMENTS FROM DAGSHUB MLflow
# --------------------------------------------------------------------
st.header("📊 Experiments from MLflow")

headers = {}
if DAGSHUB_TOKEN:
    headers["Authorization"] = f"token {DAGSHUB_TOKEN}"

exp_list_url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list"
resp = requests.get(exp_list_url, headers=headers)

if resp.status_code == 200:
    exp_data = resp.json().get("experiments", [])
    st.write(pd.DataFrame(exp_data))
else:
    st.error(f"Couldn't fetch experiments. Status code: {resp.status_code}")

# --------------------------------------------------------------------
# 2. LOAD MODEL FROM DVC (STREAMED)
# --------------------------------------------------------------------
st.header("🤖 Load Model")

MODEL_PATH = "model.joblib"   # put the real tracked path

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully from DagsHub (via streaming).")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# --------------------------------------------------------------------
# 3. PREDICTION UI
# --------------------------------------------------------------------
st.header("🔮 Make a Prediction")

cpu_request = st.number_input("CPU Request", value=0.5)
mem_request = st.number_input("Memory Request", value=256.0)
cpu_limit = st.number_input("CPU Limit", value=1.0)
mem_limit = st.number_input("Memory Limit", value=512.0)
runtime_minutes = st.number_input("Runtime (minutes)", value=60)

controller_kind = st.selectbox("Controller Type", 
                               ["deployment", "statefulset", "daemonset"])

# Encode categorical variable
encoding_map = {
    "deployment": 0,
    "statefulset": 1,
    "daemonset": 2
}

controller_encoded = encoding_map[controller_kind]

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded")
    else:
        X = [[
            cpu_request,
            mem_request,
            cpu_limit,
            mem_limit,
            runtime_minutes,
            controller_encoded
        ]]

        try:
            pred = model.predict(X)[0]
            st.success(f"Predicted CPU usage: {float(pred):.4f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
