# src/train.py
import os
import mlflow
import mlflow.sklearn
import yaml
import pandas as pd
from model_utils import load_data, train_and_save
import joblib

# load params
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# point MLflow to DagsHub tracking server for your repo
# Replace USER/REPO appropriately
DAGSHUB_USER = os.getenv('DAGSHUB_USER', 'Arjun-R-max')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO', 'cpu-usage-predictor')
mlflow.set_tracking_uri(f'https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow')

# optional: set experiment name
mlflow.set_experiment('cpu-usage-regression')

def main():
    df = load_data('data/cpu_usage.csv')

    with mlflow.start_run() as run:
        # log params
        mlflow.log_params(params['model'])
        # train
        model = train_and_save(df, params, model_out='model.joblib')
        # log model artifact
        mlflow.log_artifact('model.joblib', artifact_path='models')
        # evaluate on simple holdout
        X = df[['cpu_request','mem_request','cpu_limit','mem_limit','runtime_minutes','controller_kind']]
        y = df['cpu_usage']
        preds = model.predict(X)
        import numpy as np
        mse = float(((preds - y) ** 2).mean())
        mae = float(abs(preds - y).mean())
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('mae', mae)

        print("Run id:", run.info.run_id)
        print("MSE:", mse, "MAE:", mae)

if __name__ == '__main__':
    main()
