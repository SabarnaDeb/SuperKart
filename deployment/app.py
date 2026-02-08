import os

import joblib
import pandas as pd
from fastapi import FastAPI
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

MODEL_REPO = os.environ.get("MODEL_REPO", "SabarnaDeb/superkart-sales-rf")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "superkart_best_model.joblib")

# If model repo is private, set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in Space secrets
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=token)
model = joblib.load(model_path)

app = FastAPI(title="SuperKart Sales Predictor")


class SuperKartInput(BaseModel):
    Product_Weight: float
    Product_Sugar_Content: str
    Product_Allocated_Area: float
    Product_Type: str
    Product_MRP: float
    Store_Size: str
    Store_Location_City_Type: str
    Store_Type: str
    Store_Age: float


@app.get("/")
def home():
    return {"status": "ok", "message": "SuperKart Sales Prediction API is running."}


@app.post("/predict")
def predict(payload: SuperKartInput):
    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    df = pd.DataFrame([data])
    pred = float(model.predict(df)[0])
    return {"prediction": pred, "model_repo": MODEL_REPO}
