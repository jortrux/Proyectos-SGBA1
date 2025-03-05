import mlflow
import requests
import json

# Configurar la URI de seguimiento de MLflow en Dagshub
MLFLOW_TRACKING_URI = "https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow"
AUTH_USER = "auditoria.SGBA1"
AUTH_TOKEN = "993b92c1e06e9737e8653be41dc9193b518d8fa5"
mlflow.set_tracking_uri(f"https://{AUTH_USER}:{AUTH_TOKEN}@dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Seleccionar el modelo
MODEL_NAME = "XGBClassifier"  # Sustituye con el nombre del modelo
MODEL_VERSION = "1"  # Sustituye con la versión del modelo

# Endpoint de predicción en MLflow
PREDICTION_URL = f"{MLFLOW_TRACKING_URI}/model-versions/get-artifact?name={MODEL_NAME}&version={MODEL_VERSION}"

# Datos de entrada para la predicción
input_data = {
    "data": [[1.2, 3.4, 5.6, 7.8]]  # Sustituye con los datos reales en el formato esperado por el modelo
}

# Enviar solicitud POST con autenticación
headers = {"Content-Type": "application/json"}
response = requests.post(PREDICTION_URL, auth=(AUTH_USER, AUTH_TOKEN), headers=headers, data=json.dumps(input_data))

# Mostrar resultado
if response.status_code == 200:
    print("Predicción recibida:", response.json())
else:
    print("Error en la solicitud:", response.status_code, response.text)