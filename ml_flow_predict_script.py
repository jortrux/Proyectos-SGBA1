import pandas as pd
import mlflow.pyfunc
import dagshub

# Configurar DAGsHub con MLflow antes de cargar el modelo
dagshub.init(repo_owner='auditoria.SGBA1', repo_name='SGBA1-smartgrids', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Definir el nombre del modelo registrado
model_name = "Prophet-Precio-Luz"
model_version = 2  # Puedes cambiarlo si hay una nueva versión

# URI del modelo en MLflow
model_uri = f"models:/{model_name}/{model_version}"

# Cargar el modelo Prophet desde MLflow
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Generar fechas para la primera semana de 2025
future_dates = pd.DataFrame({"ds": pd.date_range(start="2025-01-01", periods=7*24, freq='H')})

# Hacer predicciones
predictions = loaded_model.predict(future_dates)

# Mostrar los primeros valores predichos
print(predictions.head())
