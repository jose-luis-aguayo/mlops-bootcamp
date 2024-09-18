import pickle
import re
from pathlib import Path
import pandas as pd
from train_pipeline.conf import DATASET_PATH, MODEL
from train_pipeline.functions import load_model, get_n_future_days, get_forecasting, feature_engineering

MAX_NUMBER_OF_DAYS_ALLOWED = 15

def predict_pipeline(number_of_days):
    if number_of_days > MAX_NUMBER_OF_DAYS_ALLOWED:
        return pd.DataFrame({"error":"error"}, index=[0])

    preprocessed_df = pd.read_csv(DATASET_PATH)
    df = feature_engineering(preprocessed_df)

    model = load_model(MODEL)
    dates_to_forecast = get_n_future_days(DF=df, n_days=15)
    forecasting = get_forecasting(DF=df, future_DF=dates_to_forecast, model=model)
    return  forecasting