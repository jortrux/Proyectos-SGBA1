import os
import subprocess
import pandas as pd
import numpy as np
import mlflow
import dagshub
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.logging import get_run_logger
from datetime import timedelta

# Variables de entorno
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
REPO_DATA_DIR_PATH = os.getenv("REPO_DATA_DIR_PATH")

# Ruta de archivo
FILE_PATH_HOGAR = (
    REPO_DATA_DIR_PATH + "/data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
    if REPO_DATA_DIR_PATH
    else "../../../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
)

def run_command(command: str) -> str:
    # Ejecuta un comando en la terminal
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error ejecutando '{command}': {e.stderr}")

@task(name="authenticate_dagshub", retries=2)
def authenticate_dagshub():
    logger = get_run_logger()
    try:
        token = os.getenv("DAGSHUB_TOKEN")
        if not token:
            raise ValueError("Falta DAGSHUB_TOKEN.")
        dagshub.auth.add_app_token(token=token)
        logger.info("Autenticación con Dagshub OK.")
        return True
    except Exception as e:
        logger.error(f"Fallo autenticando Dagshub: {e}")
        raise

@task(name="initialize_dagshub")
def initialize_dagshub():
    logger = get_run_logger()
    try:
        if not all([DAGSHUB_USERNAME, DAGSHUB_REPO_NAME]):
            raise ValueError("Falta DAGSHUB_USERNAME o DAGSHUB_REPO_NAME.")
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
        logger.info("Dagshub inicializado.")
        return True
    except Exception as e:
        logger.error(f"Fallo al inicializar Dagshub: {e}")
        raise

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(file_path):
    # Carga y preprocesa datos
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_prophet = df[["timestamp", "consumo_kwh"]].rename(columns={"timestamp": "ds", "consumo_kwh": "y"})
    train = df_prophet[df_prophet["ds"] < "2020-01-01"]
    test = df_prophet[df_prophet["ds"] >= "2020-01-01"]
    return train, test

@task
def define_holidays():
    # Festivos de Barcelona
    festivos = pd.DataFrame({
        "holiday": "festivo",
        "ds": pd.to_datetime([
            "2017-01-06","2017-04-14","2017-04-17","2017-05-01","2017-06-05",
            "2017-06-24","2017-08-15","2017-09-11","2017-09-25","2017-10-12",
            "2017-11-01","2017-12-06","2017-12-08","2017-12-25","2017-12-26",
            "2018-01-01","2018-01-06","2018-03-30","2018-04-02","2018-05-01",
            "2018-08-15","2018-09-11","2018-10-12","2018-11-01","2018-12-06",
            "2018-12-08","2018-12-25","2018-12-26","2019-01-01","2019-04-19",
            "2019-04-22","2019-05-01","2019-06-10","2019-06-24","2019-08-15",
            "2019-09-11","2019-09-24","2019-10-12","2019-11-01","2019-12-06",
            "2019-12-25","2019-12-26","2020-01-01","2020-01-06","2020-04-10",
            "2020-04-13","2020-05-01","2020-06-01","2020-06-24","2020-08-15",
            "2020-09-11","2020-09-24","2020-10-12","2020-12-08","2020-12-25",
            "2020-12-26"
        ]),
        "lower_window": 0,
        "upper_window": 1
    })
    return festivos

@task
def build_model(train, festivos):
    # Crea y entrena Prophet
    model = Prophet(
        changepoint_prior_scale=0.5,
        seasonality_mode="multiplicative",
        holidays=festivos
    )
    model.add_seasonality(name="trimestral", period=90.25, fourier_order=10)
    model.add_seasonality(name="mensual", period=30.5, fourier_order=10)
    model.add_seasonality(name="semanal", period=7, fourier_order=10)
    model.fit(train)
    return model

@task
def sequential_validation(model, train, test):
    # Valida día a día en 2020
    rmse_values = []
    mae_values = []
    preds_total = pd.DataFrame()
    errores_previos = []

    for day in pd.date_range(start="2020-01-01", end=test["ds"].max(), freq="D"):
        test_day = test[(test["ds"] >= day) & (test["ds"] < day + pd.Timedelta(days=1))]
        if test_day.empty:
            continue
        future = pd.DataFrame({"ds": test_day["ds"]})
        fcst = model.predict(future)

        if len(errores_previos) > 10:
            ajuste_error = np.mean(errores_previos[-10:])
        else:
            ajuste_error = 0

        fcst["yhat_ajustado"] = fcst["yhat"] + ajuste_error
        rmse = np.sqrt(mean_squared_error(test_day["y"], fcst["yhat_ajustado"]))
        mae = mean_absolute_error(test_day["y"], fcst["yhat_ajustado"])
        rmse_values.append(rmse)
        mae_values.append(mae)

        errores_previos.extend(test_day["y"].values - fcst["yhat"].values)
        fcst["real"] = test_day["y"].values
        preds_total = pd.concat([preds_total, fcst[["ds","yhat","yhat_ajustado","real"]]])

    final_rmse = np.mean(rmse_values)
    final_mae = np.mean(mae_values)
    return final_rmse, final_mae, preds_total

@task
def plot_results(preds_total):
    # Grafica
    plt.figure(figsize=(26, 8))
    plt.plot(preds_total["ds"], preds_total["real"], label="Real", alpha=0.6)
    plt.plot(preds_total["ds"], preds_total["yhat"], label="Predicción", linestyle="dashed")
    plt.xlabel("Fecha")
    plt.ylabel("Consumo (kWh)")
    plt.title("Comparación Consumo Real vs Predicción (Validación Secuencial)")
    plt.legend()
    plt.grid(True)
    plot_path = "prophet_predictions.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# MLflow Wrapper
class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict(self, context, model_input):
        df_input = pd.DataFrame(model_input)
        df_input.columns = ["ds"]
        fcst = self.model.predict(df_input)
        return fcst[["ds","yhat"]]

@task
def register_model(model, rmse, mae, plot_path):
    # Registra en MLflow
    mlflow.set_experiment("prophet_consumo_individual_basico")
    with mlflow.start_run(run_name="prophet_consumo_individual_basico") as run:
        run_id = run.info.run_id
        mlflow.log_param("changepoint_prior_scale", 0.5)
        mlflow.log_param("seasonality_mode", "multiplicative")
        mlflow.log_param("Trimestral_Period", 90.25)
        mlflow.log_param("Mensual_Period", 30.5)
        mlflow.log_param("Semanal_Period", 7)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_artifact(plot_path)

        model_name = "Prophet-Consumo-Hogar"
        model_uri = f"runs:/{run_id}/prophet_model"
        mlflow.pyfunc.log_model(artifact_path="prophet_model", python_model=ProphetWrapper(model))
        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Modelo registrado. Run ID: {run_id}")

@flow(name="prophet_consumo_flow", log_prints=True)
def prophet_consumo_flow(file_path: str = FILE_PATH_HOGAR):
    authenticate_dagshub()
    initialize_dagshub()
    train, test = load_data(file_path)
    festivos = define_holidays()
    model = build_model(train, festivos)
    rmse, mae, preds_total = sequential_validation(model, train, test)
    plot_path = plot_results(preds_total)
    register_model(model, rmse, mae, plot_path)
    print(f"RMSE Final: {rmse:.2f}, MAE Final: {mae:.2f}")

if __name__ == "__main__":
    prophet_consumo_flow()
