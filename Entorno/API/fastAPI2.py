from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import dagshub
from typing import List

# Configuración de DAGsHub con MLflow
mlflow.set_tracking_uri("https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Nombre del modelo, esto hay que cambiarlo luego cuando tengamos varios modelos, para que se pueda elegir
model_name = "Prophet-Precio-Luz"
model_version = 2  # Versión, también hay que ajustarlo a distintas versiones de modelos

# URI del modelo en MLflow
model_uri = f"models:/{model_name}/{model_version}"

# Carga del modelo 
try:
    loaded_model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    loaded_model = None

# Iniciar FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "API de predicción con MLflow y FastAPI"}

# Definir el esquema de entrada
class PredictionInput(BaseModel):
    ds: List[str]  # Lista de fechas en formato string

@app.post("/predict")
def predict(data: PredictionInput):
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Convertir la lista de fechas en un DataFrame con la columna "ds"
        df = pd.DataFrame({"ds": data.ds})
        predictions = loaded_model.predict(df)

        # Convertir a JSON
        return {"prediction": predictions.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")
