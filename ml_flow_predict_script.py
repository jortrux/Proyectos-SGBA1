import mlflow
import mlflow.pyfunc
import pandas as pd
import dagshub
import os

try:
    # Inicializar DagsHub
    dagshub.init(repo_owner='auditoria.SGBA1', repo_name='SGBA1-smartgrids', mlflow=True)
    print("DagsHub inicializado")

    mlflow.set_tracking_uri("https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")
    
    model_uri = "models:/XGBClassifier/1"

    mlflow.pyfunc.get_model_dependencies(model_uri) 
    model = mlflow.pyfunc.load_model("models:/XGBClassifier/1")

    print("Modelo cargado")

    # Datos de ejemplo para la predicción (ajusta según tus características)
    input_data = pd.DataFrame({
        "feature1": [0.5],
        "feature2": [1.2],
        "feature3": [0.3],
        "feature4": [0.8],
        "feature5": [1.5],
        "feature6": [0.7],
        "feature7": [0.2],
        "feature8": [1.0],
        "feature9": [0.4],
        "feature10": [0.9]
    })

    # Realizar la predicción
    prediction = model.predict(input_data)
    print("Predicción:", prediction)

except mlflow.exceptions.MlflowException as e:
    print("Error al cargar el modelo o realizar la predicción:", e)
except Exception as e:
    print("Ocurrió un error:", e)