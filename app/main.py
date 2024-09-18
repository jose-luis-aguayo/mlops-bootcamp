from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.model.model import predict_pipeline
import json

app = FastAPI()

def parse_to_json(df):
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    return parsed

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": ""}

@app.post("/predict")
def predict():
    #TODO: add validation max_days = 15
    forecast = predict_pipeline()
    response = (parse_to_json(forecast
                             .astype({"date": str})
                             [["date","Predicted_Income"]]))
    return JSONResponse(content=response)