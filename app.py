from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict, Any, List


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class PredictRequest(BaseModel):
    features: Dict[str, float]


class PredictBatchRequest(BaseModel):
    instances: List[Dict[str, float]]


app = FastAPI(title="TDSE Logistic Regression API")


def load_model(path="best_model.npz"):
    data = np.load(path, allow_pickle=True)
    w = data["w"]
    b = float(data["b"])
    mu = data["mu"]
    sigma = data["sigma"]
    features = data["features"]
    try:
        features = features.tolist()
    except Exception:
        pass
    return {"w": w, "b": b, "mu": mu, "sigma": sigma, "features": features}
MODEL = None


@app.on_event("startup")
def startup_event():
    global MODEL
    MODEL = load_model()


@app.post("/predict")
def predict(req: PredictRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    features_order = list(MODEL["features"])
    try:
        x = np.array([req.features[f] for f in features_order], dtype=float)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    mu = MODEL["mu"]
    sigma = MODEL["sigma"]
    x_norm = (x - mu) / (sigma + 1e-8)
    x_norm = x_norm.reshape(1, -1)
    w = MODEL["w"]
    b = MODEL["b"]
    prob = float(sigmoid(x_norm @ w + b).squeeze())
    pred = int(prob >= 0.5)
    return {"probability": prob, "prediction": pred}

@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest):
    global MODEL
    if MODEL is None:raise HTTPException(status_code=500, detail="Model not loaded")
    features_order = list(MODEL["features"])
    results = []
    mu = MODEL["mu"]
    sigma = MODEL["sigma"]
    w = MODEL["w"]
    b = MODEL["b"]
    for i, inst in enumerate(req.instances):
        try:
            x = np.array([inst[f] for f in features_order], dtype=float)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
        x_norm = (x - mu) / (sigma + 1e-8)
        x_norm = x_norm.reshape(1, -1)
        prob = float(sigmoid(x_norm @ w + b).squeeze())
        pred = int(prob >= 0.5)
        results.append({"probability": prob, "prediction": pred})
    return {"predictions": results}
