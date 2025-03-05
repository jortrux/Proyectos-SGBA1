from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.pyfunc
import dagshub

# Configurar DAGsHub con MLflow antes de cargar el modelo
dagshub.init(repo_owner='auditoria.SGBA1', repo_name='SGBA1-smartgrids', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Definir el nombre del modelo registrado
model_name = "Prophet-Precio-Luz"
model_version = 2  # Ajustar versión si es necesario

# URI del modelo en MLflow
model_uri = f"models:/{model_name}/{model_version}"

# Cargar el modelo Prophet desde MLflow
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

@app.post("/predict")
def predict(data: dict):
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Convertir los datos de entrada a DataFrame
        df = pd.DataFrame([data])

        # Hacer predicción
        predictions = loaded_model.predict(df)

        # Convertir a JSON
        return {"prediction": predictions.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")
    

    