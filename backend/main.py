import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from backend.schemas import HouseInput, PredictionOut

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "pipeline.joblib"
DB_PATH = BASE_DIR / "backend" / "history.db"

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

app = FastAPI(title="Immo Estimator")
model = joblib.load(MODEL_PATH)


def get_conn():
    return sqlite3.connect(DB_PATH)


with get_conn() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            inputs TEXT NOT NULL,
            predicted_price REAL NOT NULL
        )
    """)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(house: HouseInput):
    df = pd.DataFrame([house.model_dump()])[FEATURES]
    price = float(model.predict(df)[0]) * 100_000   # la cible est en centaines de milliers de $

    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO history (inputs, predicted_price) VALUES (?, ?)",
                (json.dumps(house.model_dump()), price),
            )
    finally:
        conn.close()

    return {"predicted_price": round(price, 2)}


@app.get("/history")
def history(limit: int = 20):
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, created_at, inputs, predicted_price FROM history "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()

    return [
        {"id": r[0], "created_at": r[1], "inputs": json.loads(r[2]), "predicted_price": r[3]}
        for r in rows
    ]