import pickle
import re
from pathlib import Path

__version__ = "1.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/forecasting_income_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_pipeline(payload):
    forecast = model.predict([payload])
    return forecast[0]