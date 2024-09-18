from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline

app = FastAPI()

class Request(BaseModel):
    Input: str

class Response(BaseModel):
    forecast: str

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": ""}

@app.post("/predict", response_model=Response)
def predict(payload: Request):
    forecast = predict_pipeline(payload.Input)
    return {"forecast": forecast}    