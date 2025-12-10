import streamlit as st
import joblib
import os
import requests
import mlflow

st.title("CPU Usage Model Dashboard")

# Directly load model stored in the GitHub repo
MODEL_PATH = "model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully from GitHub!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# MLflow (optional)
MLFLOW_URI = "https://dagshub.com/Arjun-R-max/cpu-usage-predictor.mlflow"

st.markdown("### Experiments (Public MLflow API)")

try:
    resp = requests.get(f"{MLFLOW_URI}/api/2.0/mlflow/experiments/list")
    st.json(resp.json())
except:
    st.warning("Cannot fetch MLflow experiments")

# Prediction UI
st.markdown("## Predict CPU Usage")

cpu_request = st.number_input("cpu_request", value=0.5)
mem_request = st.number_input("mem_request", value=256.0)
cpu_limit = st.number_input("cpu_limit", value=1.0)
mem_limit = st.number_input("mem_limit", value=512.0)
runtime_minutes = st.number_input("runtime_minutes", value=60)
controller_kind = st.selectbox(
    "controller_kind", ["deployment", "statefulset", "daemonset"]
)

if st.button("Predict"):
    if model:
        X = [[cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_kind]]
        pred = model.predict(X)[0]
        st.success(f"Predicted CPU usage: {float(pred)}")
    else:
        st.error("Model not loaded")
