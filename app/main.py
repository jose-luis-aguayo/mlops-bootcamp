from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": ""}