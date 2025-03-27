import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.xgboost
import dagshub
from typing import List, Union

# Configurar autenticación para DAGsHub y MLflow desde variables de entorno o por defecto
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "auditoria.SGBA1")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "ee9be1f2d99f10b3647e4bccee075e65178ecf03")

# Configurar MLflow con DAGsHub
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USERNAME}/Proyectos-SGBA1.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

dagshub.auth.add_app_token(DAGSHUB_TOKEN)
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name="Proyectos-SGBA1", mlflow=True)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Modelos disponibles
MODELOS_DISPONIBLES = {
    "consumo_hogar": "XGBoost-Consumo-Hogar"
}

# Caché para modelos
model_cache = {}

def obtener_ultima_version_modelo(model_name: str) -> int:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["None", "Production"])
        return int(versions[0].version)
    except Exception as e:
        raise RuntimeError(f"Error obteniendo la última versión del modelo '{model_name}': {e}")

def cargar_modelo(model_key: str):
    if model_key not in MODELOS_DISPONIBLES:
        raise ValueError(f"Modelo no reconocido: {model_key}")

    model_name = MODELOS_DISPONIBLES[model_key]
    model_version = obtener_ultima_version_modelo(model_name)
    model_uri = f"models:/{model_name}/{model_version}"

    try:
        if model_key not in model_cache:
            print(f"Cargando modelo '{model_name}' versión {model_version}...")
            # Este modelo es XGBoost puro
            model_cache[model_key] = mlflow.xgboost.load_model(model_uri)
        return model_cache[model_key]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo '{model_name}': {e}")

# FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API de predicción para Consumo Hogar usando XGBoost en Kubernetes"}

class PredictionInput(BaseModel):
    model_type: str
    features: List[dict]  # Cada dict es una fila con sus columnas

@app.post("/predict")
def predict(data: PredictionInput):
    if data.model_type not in MODELOS_DISPONIBLES:
        raise HTTPException(status_code=400, detail=f"Modelo no válido. Opciones: {list(MODELOS_DISPONIBLES.keys())}")

    modelo = cargar_modelo(data.model_type)

    try:
        df = pd.DataFrame(data.features)
        prediction = modelo.predict(df)
        return {"model": data.model_type, "prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")
