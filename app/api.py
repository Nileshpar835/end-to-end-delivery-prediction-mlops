from fastapi import FastAPI
import torch
import pandas as pd
import joblib
import os

from model.model import get_nn_model

app = FastAPI()

# Absolute paths (IMPORTANT)
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")

# Load columns first
columns = joblib.load(COLUMNS_PATH)

# 🔥 THIS IS THE REAL FIX
input_dim = len(columns)

print(f"Loaded input_dim from training: {input_dim}")

# Load model correctly
model = get_nn_model(input_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


@app.get("/")
def home():
    return {"message": "API is working 🚀"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Same preprocessing as training
    df = pd.get_dummies(df)

    # Align columns EXACTLY
    df = df.reindex(columns=columns, fill_value=0)
    df = df.astype(float)   # 🔥 FIX

    X = torch.tensor(df.values, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X).item()

    return {"prediction": round(pred, 2)}