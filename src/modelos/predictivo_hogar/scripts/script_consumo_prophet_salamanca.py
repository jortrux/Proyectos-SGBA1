import os
import pandas as pd
import numpy as np
import mlflow
import dagshub
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.logging import get_run_logger
import subprocess
from datetime import timedelta


DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
REPO_DATA_DIR_PATH = os.getenv('REPO_DATA_DIR_PATH')


def run_command(command: str) -> str:
    """Ejecuta un comando en la terminal y maneja errores"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error ejecutando '{command}': {e.stderr}")

@task(name="authenticate_dagshub", retries=2)
def authenticate_dagshub():
    """Task to authenticate with DagsHub using environment variables."""
    logger = get_run_logger()
    try:
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        if not dagshub_token:
            raise ValueError("DAGSHUB_TOKEN environment variable not set")
        
        dagshub.auth.add_app_token(token=dagshub_token)
        logger.info("Successfully authenticated with DagsHub")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with DagsHub: {str(e)}")
        raise

@task(name="initialize_dagshub")
def initialize_dagshub():
    """Task to initialize DagsHub connection."""
    logger = get_run_logger()
    try:
        if not all([DAGSHUB_USERNAME, DAGSHUB_REPO_NAME]):
            raise ValueError("Missing required environment variables: DAGSHUB_USERNAME or DAGSHUB_REPO_NAME")
        
        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True
        )
        logger.info(f"Successfully initialized DagsHub for repo: {DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DagsHub: {str(e)}")
        raise

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(file_path):
    """Cargar y preprocesar los datos"""
    df = pd.read_csv(file_path)
    
    # Preprocesamiento
    df.rename(columns={'datetime': 'ds', 'Consumo_kWh': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df = df.sort_values(by='ds')
    df['y'] = df['y'].interpolate(method='linear')
    
    # Variables adicionales
    df['weekend'] = df['ds'].dt.weekday.isin([5, 6]).astype(int)
    df['day_of_week'] = df['ds'].dt.weekday
    df['month'] = df['ds'].dt.month
    
    # DataFrame de festivos
    df_festivos = df[df['es_festivo'] == 1][['ds']]
    
    return df, df_festivos

@task
def split_data(df):
    """Dividir datos en entrenamiento y prueba"""
    split_date = df['ds'].max() - pd.Timedelta(days=7)
    train = df[df['ds'] <= split_date]
    test = df[df['ds'] > split_date]
    return train, test

@task
def train_prophet_model(train, df_festivos):
    """Entrenar modelo Prophet"""
    model = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.03,
        holidays_prior_scale=10
    )
    
    # Añadir estacionalidades
    model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=15)
    
    holidays = pd.DataFrame({
        'ds': df_festivos['ds'],
        'holiday': 'festivo'
    })
    
    model.add_seasonality(name='festivos', period=365, fourier_order=10)
    model.add_country_holidays(country_name='ES')
    
    # Añadir regresores
    model.add_regressor('weekend')
    model.add_regressor('day_of_week')
    model.add_regressor('month')
    
    # Ajustar modelo
    model.fit(train)
    
    return model

@task
def evaluate_model(model, test):
    """Evaluar rendimiento del modelo"""
    future = test[['ds', 'weekend', 'day_of_week', 'month']]
    forecast = model.predict(future)
    
    y_true = test['y'].values
    y_pred = forecast['yhat'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return mae, rmse, test, forecast

@task
def plot_results(test, y_true, y_pred):
    """Graficar resultados"""
    plt.figure(figsize=(10, 5))
    plt.plot(test['ds'], y_true, label="Real", marker='o')
    plt.plot(test['ds'], y_pred, label="Predicción", marker='x')
    plt.legend()
    plt.title("Validación del modelo Prophet")
    plt.xlabel("Fecha")
    plt.ylabel("Consumo hogar individual")
    plt.xticks(rotation=45)
    
    validation_plot_path = "validacion_prediccion.png"
    plt.savefig(validation_plot_path)
    plt.close()
    
    return validation_plot_path

@task
def register_model_mlflow(model, mae, rmse, validation_plot_path):
    """Registrar modelo en MLflow"""
    class ProphetWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model
        def predict(self, context, model_input):
            model_input = pd.DataFrame(model_input)
            model_input.columns = ['ds']
            forecast = self.model.predict(model_input)
            return forecast[['ds', 'yhat']]
    
    # Configurar DAGsHub y MLflow
    mlflow.set_experiment("Predicción Consumo Hogar con Prophet")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Registrar parámetros y métricas
        mlflow.log_param("modelo", "Prophet")
        mlflow.log_param("periodo_validacion_dias", 7)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        
        # Subir imagen de validación
        mlflow.log_artifact(validation_plot_path)
        
        # Registrar modelo
        model_name = "Prophet-Consumo-Hogar"
        model_uri = f"runs:/{run_id}/prophet_model"
        mlflow.pyfunc.log_model(artifact_path="prophet_model", python_model=ProphetWrapper(model))
        mlflow.register_model(model_uri=model_uri, name=model_name)
        
        print(f"¡Modelo registrado en DAGsHub y MLflow correctamente con Run ID: {run_id}!")

@flow(log_prints=True)
def prophet_workflow(file_path):
    """Flujo principal de trabajo"""
    # Cargar datos
    df, df_festivos = load_data(file_path)
    
    # Dividir datos
    train, test = split_data(df)
    
    # Entrenar modelo
    model = train_prophet_model(train, df_festivos)
    
    # Evaluar modelo
    mae, rmse, test_data, forecast = evaluate_model(model, test)
    
    # Graficar resultados
    validation_plot_path = plot_results(test_data, test_data['y'].values, forecast['yhat'].values)
    
    # Registrar modelo en MLflow
    authenticate_dagshub()
    initialize_dagshub()
    register_model_mlflow(model, mae, rmse, validation_plot_path)

if __name__ == "__main__":
    # Ruta del archivo de datos
    file_path = "data/processed/predictivo_hogar/consumo_salamanca_cleaned.csv"
    
    # Ejecutar flujo de trabajo
    prophet_workflow(file_path)