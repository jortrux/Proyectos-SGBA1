import mlflow
import mlflow.sklearn

DAGSHUB_USERNAME = "auditoria.SGBA1"
DAGSHUB_TOKEN = "ee9be1f2d99f10b3647e4bccee075e65178ecf03"

mlflow.set_tracking_uri(f"https://{DAGSHUB_USERNAME}:{DAGSHUB_TOKEN}@dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Buscar el último run_id donde se guardó el modelo
runs = mlflow.search_runs()
latest_run_id = runs.iloc[0]["run_id"]  # Obtener el último experimento

# Cargar modelo usando run_id
model_uri = f"runs:/{latest_run_id}/iris_model"
model = mlflow.sklearn.load_model(model_uri)

# Datos hardcodeados (Iris)
import numpy as np
data = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(data)

print(f"Predicción: {prediction} -> Clase: {['Setosa', 'Versicolor', 'Virginica'][prediction[0]]}")

