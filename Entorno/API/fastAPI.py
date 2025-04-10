import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.pyfunc
import dagshub
from typing import List

# Configurar DAGsHub y MLflow
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "auditoria.SGBA1")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "ee9be1f2d99f10b3647e4bccee075e65178ecf03")

os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USERNAME}/Proyectos-SGBA1.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

dagshub.auth.add_app_token(DAGSHUB_TOKEN)
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name="Proyectos-SGBA1", mlflow=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Modelos disponibles
MODELOS_DISPONIBLES = {
    "consumo_hogar": "XGBoost-Consumo-Hogar"
}

model_cache = {}

# Cargar modelo desde DAGsHub
def cargar_modelo(model_key: str):
    if model_key not in MODELOS_DISPONIBLES:
        raise HTTPException(status_code=400, detail="Modelo no reconocido.")

    model_name = MODELOS_DISPONIBLES[model_key]
    model_uri = f"models:/{model_name}/latest"

    try:
        if model_key not in model_cache:
            model_cache[model_key] = mlflow.pyfunc.load_model(model_uri)
        return model_cache[model_key]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {e}")

# FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API de predicción para modelos MLflow desde DAGsHub"}

# Entrada esperada
class PredictionInput(BaseModel):
    model_type: str
    ds: List[str]  # Lista de timestamps

@app.post("/predict")
def predict(data: PredictionInput):
    modelo = cargar_modelo(data.model_type)

    try:
        df = pd.DataFrame({"ds": pd.to_datetime(data.ds)})

        df["hour"] = df["ds"].dt.hour
        df["weekday"] = df["ds"].dt.weekday
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
        df["is_holiday"] = 0  
        df["tmed"] = 17.0
        df["tmin"] = 12.0
        df["tmax"] = 22.0
        df["prec"] = 0.0
        df["velmedia"] = 2.5
        df["racha"] = 4.1
        df["sol"] = 6.2
        df["hrMedia"] = 68.0
        df["año"] = df["ds"].dt.year

        df = df.drop(columns=["ds"])


        forecast = modelo.predict(df)

        # Para poder devolver la predicción con sus fechas originales:
        return {
            "model": data.model_type,
            "prediction": [
                {"ds": ts, "yhat": yhat} for ts, yhat in zip(data.ds, forecast.tolist())
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")
