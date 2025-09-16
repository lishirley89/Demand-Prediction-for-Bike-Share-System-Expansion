from fastapi import FastAPI
from pydantic import BaseModel
from code.inference import predict_for_point

app = FastAPI()

class PredictRequest(BaseModel):
    lat: float
    lng: float
    year: int = 2025

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    df = predict_for_point(req.lat, req.lng, req.year)
    return df.to_dict(orient="records")