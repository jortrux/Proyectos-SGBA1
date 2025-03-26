from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

# Cargar modelo desde Dagshub
MODEL_URI = "https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow/models:/1"
modelo = mlflow.pyfunc.load_model(MODEL_URI)

@app.post("/predict")
async def predict(data: dict):
    X = data["inputs"]
    predicciones = modelo.predict(X).tolist()
    return {"predictions": predicciones}
