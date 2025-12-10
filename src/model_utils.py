# src/model_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

FEATURES = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
TARGET = 'cpu_usage'

def load_data(path):
    df = pd.read_csv(path)
    return df

def make_pipeline(params):
    # controller_kind is categorical
    cat_cols = ['controller_kind']
    num_cols = [c for c in FEATURES if c not in cat_cols]

    preproc = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ], remainder='passthrough')

    model = RandomForestRegressor(n_estimators=params['model']['n_estimators'],
                                  max_depth=params['model']['max_depth'],
                                  random_state=params['train']['random_state'])
    pipe = Pipeline([('preproc', preproc), ('model', model)])
    return pipe

def train_and_save(df, params, model_out='model.joblib'):
    X = df[FEATURES]
    y = df[TARGET]

    pipe = make_pipeline(params)
    pipe.fit(X, y)
    joblib.dump(pipe, model_out)
    return pipe
