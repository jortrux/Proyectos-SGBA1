import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import dagshub
import matplotlib.pyplot as plt
import pickle
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configurar DAGsHub con MLflow
dagshub.init(repo_owner='auditoria.SGBA1', repo_name='SGBA1-smartgrids', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")
mlflow.set_experiment("Predicción Precio Luz con Prophet")

# Cargar los datos
file_path = "datos_completos_limpios.csv"  # Asegúrate de que la ruta sea correcta
df = pd.read_csv(file_path)

# Preparar los datos para Prophet
df['datetime'] = pd.to_datetime(df['datetime'])
df_prophet = df[['datetime', 'precio']].rename(columns={'datetime': 'ds', 'precio': 'y'})

# Definir el punto de corte para la validación (últimos 14 días)
split_date = df_prophet['ds'].max() - pd.Timedelta(days=14)
train = df_prophet[df_prophet['ds'] <= split_date]
test = df_prophet[df_prophet['ds'] > split_date]

# Clase personalizada para Prophet en MLflow
class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        model_input = pd.DataFrame(model_input)
        model_input.columns = ['ds']
        forecast = self.model.predict(model_input)
        return forecast[['ds', 'yhat']]

# Iniciar experimento en MLflow
with mlflow.start_run() as run:
    run_id = run.info.run_id  # Guardamos el Run ID

    # Registrar parámetros
    mlflow.log_param("modelo", "Prophet")
    mlflow.log_param("periodo_validacion_dias", 14)

    # Crear y ajustar el modelo Prophet
    model = Prophet()
    model.fit(train)

    # Hacer predicciones para el conjunto de validación
    future = test[['ds']]  # Usamos las fechas de prueba
    forecast = model.predict(future)

    # Evaluar el modelo
    y_true = test['y'].values
    y_pred = forecast['yhat'].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Registrar métricas en MLflow
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.plot(test['ds'], y_true, label="Real", marker='o')
    plt.plot(test['ds'], y_pred, label="Predicción", marker='x')
    plt.legend()
    plt.title("Validación del modelo Prophet")
    plt.xlabel("Fecha")
    plt.ylabel("Precio de la luz")
    plt.xticks(rotation=45)

    # Guardar la gráfica y subirla a MLflow
    plt.savefig("validacion_prediccion.png")
    mlflow.log_artifact("validacion_prediccion.png")
    plt.show()

    # Guardar el modelo Prophet como un archivo pickle
    model_path = "prophet_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Registrar el modelo en MLflow usando la clase personalizada
    model_name = "Prophet-Precio-Luz"
    model_uri = f"runs:/{run_id}/prophet_model"

    mlflow.pyfunc.log_model(artifact_path="prophet_model", python_model=ProphetWrapper(model))

    # Registrar el modelo en MLflow Model Registry
    mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"¡Modelo registrado en DAGsHub y MLflow correctamente con Run ID: {run_id}!")
