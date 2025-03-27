import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import dagshub
from typing import List

# Configurar credenciales de DAGsHub desde variables de entorno
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "auditoria.SGBA1")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN", "ee9be1f2d99f10b3647e4bccee075e65178ecf03")

# Configurar autenticación para DAGsHub y MLflow
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USERNAME}/Proyectos-SGBA1.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Iniciar DAGsHub con autenticación
dagshub.auth.add_app_token(DAGSHUB_TOKEN)
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name="Proyectos-SGBA1", mlflow=True)

# Configurar MLflow con autenticación
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Modelos disponibles en DAGsHub
MODELOS_DISPONIBLES = {
    "consumo_hogar": "XGBoost-Consumo-Hogar"
}


# Caché de modelos para evitar cargas repetidas
model_cache = {}

def obtener_ultima_version_modelo(model_name: str) -> int:
    """Obtiene la última versión registrada del modelo en DAGsHub."""
    try:
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name)
        if versions:
            print(f"Versiones disponibles para el modelo {model_name}: {versions}")
            return max(int(v.version) for v in versions)  # Obtiene la versión más alta
        else:
            raise ValueError(f"No hay versiones disponibles para el modelo {model_name}")
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
            print(f"Cargando modelo '{model_name}' versión {model_version} desde MLflow...")

            if model_name.startswith("XGBoost"):
                model_cache[model_key] = mlflow.xgboost.load_model(model_uri)
            else:
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
