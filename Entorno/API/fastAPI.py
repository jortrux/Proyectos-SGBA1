from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import dagshub
from typing import List

# Configuración de DAGsHub con MLflow
dagshub.init(repo_owner='auditoria.SGBA1', repo_name='SGBA1-smartgrids', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Diccionario con los modelos disponibles y sus versiones
MODELOS_DISPONIBLES = {
    "precio_luz": {"name": "Prophet-Precio-Luz", "version": 3},
    "consumo_hogar": {"name": "Prophet-Consumo-Hogar", "version": 3}
}

# Cargar modelos en memoria (optimización)
model_cache = {}

def cargar_modelo(model_key: str):
    """Carga el modelo desde MLflow si no está en caché."""
    if model_key not in MODELOS_DISPONIBLES:
        raise ValueError(f"Modelo no reconocido: {model_key}")

    model_name = MODELOS_DISPONIBLES[model_key]["name"]
    model_version = MODELOS_DISPONIBLES[model_key]["version"]
    model_uri = f"models:/{model_name}/{model_version}"

    try:
        if model_key not in model_cache:
            model_cache[model_key] = mlflow.pyfunc.load_model(model_uri)
        return model_cache[model_key]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo '{model_name}': {e}")

# Iniciar FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API de predicción con MLflow y FastAPI. Modelos disponibles: precio_luz, consumo_hogar"}

# Definir el esquema de entrada
class PredictionInput(BaseModel):
    model_type: str  # Modelo a utilizar: "precio_luz" o "consumo_hogar"
    ds: List[str]  # Lista de fechas en formato string

@app.post("/predict")
def predict(data: PredictionInput):
    """Realiza predicciones usando el modelo seleccionado."""
    if data.model_type not in MODELOS_DISPONIBLES:
        raise HTTPException(status_code=400, detail=f"Modelo no válido. Opciones: {list(MODELOS_DISPONIBLES.keys())}")

    # Cargar modelo correspondiente
    modelo = cargar_modelo(data.model_type)

    try:
        # Convertir la lista de fechas en un DataFrame con la columna "ds"
        df = pd.DataFrame({"ds": data.ds})
        predictions = modelo.predict(df)

        # Convertir a JSON
        return {"model": data.model_type, "prediction": predictions.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")
