from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from app.model.model import predict_pipeline
import json

app = FastAPI()

class Request(BaseModel):
    number_of_days: int

def parse_to_json(df):
    res = df.to_json(orient="records")
    parsed = json.loads(res)
    return parsed

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": ""}

@app.post("/predict")
def predict(payload: Request):
    forecast = predict_pipeline(payload.number_of_days)
    
    if "error" in forecast.values:
        return {"Error": "max number of days allowed is 15"}
    
    response = (parse_to_json(forecast
                             .astype({"date": str})
                             [["date","Predicted_Income"]]))
    return JSONResponse(content=response)