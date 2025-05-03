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

FILE_PATH_HOGAR = (
    REPO_DATA_DIR_PATH + "/data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
    if REPO_DATA_DIR_PATH
    else "../../../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
)

FESTIVOS = [
    "2017-01-06","2017-04-14","2017-04-17","2017-05-01","2017-06-05","2017-06-24",
    "2017-08-15","2017-09-11","2017-09-25","2017-10-12","2017-11-01","2017-12-06",
    "2017-12-08","2017-12-25","2017-12-26","2018-01-01","2018-01-06","2018-03-30",
    "2018-04-02","2018-05-01","2018-08-15","2018-09-11","2018-10-12","2018-11-01",
    "2018-12-06","2018-12-08","2018-12-25","2018-12-26","2019-01-01","2019-04-19",
    "2019-04-22","2019-05-01","2019-06-10","2019-06-24","2019-08-15","2019-09-11",
    "2019-09-24","2019-10-12","2019-11-01","2019-12-06","2019-12-25","2019-12-26",
    "2020-01-01","2020-01-06","2020-04-10","2020-04-13","2020-05-01","2020-06-01",
    "2020-06-24","2020-08-15","2020-09-11","2020-09-24","2020-10-12","2020-12-08",
    "2020-12-25","2020-12-26"
]

regresores = [
    'hour', 'weekday', 'is_weekend', 'is_holiday',
    'tmed', 'tmin', 'tmax', 'prec', 'velmedia', 'racha', 'sol', 'hrMedia', 'year',
    'lag_1h', 'lag_24h', 'rolling_3h', 'rolling_24h', 'delta',
    'hour_sin', 'hour_cos'
]

@task(name="authenticate_dagshub", retries=2)
def authenticate_dagshub():
    logger = get_run_logger()
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise ValueError("Falta DAGSHUB_TOKEN.")
    dagshub.auth.add_app_token(token=token)
    logger.info("Autenticación con Dagshub OK.")

@task(name="initialize_dagshub")
def initialize_dagshub():
    logger = get_run_logger()
    if not all([DAGSHUB_USERNAME, DAGSHUB_REPO_NAME]):
        raise ValueError("Falta DAGSHUB_USERNAME o DAGSHUB_REPO_NAME.")
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    logger.info("Dagshub inicializado.")

@task(name="load_and_prepare_data", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = df["timestamp"].dt.strftime("%Y-%m-%d").isin(FESTIVOS).astype(int)

    df["lag_1h"] = df["consumo_kwh"].shift(1)
    df["lag_24h"] = df["consumo_kwh"].shift(24)
    df["rolling_3h"] = df["consumo_kwh"].shift(1).rolling(3).mean()
    df["rolling_24h"] = df["consumo_kwh"].shift(1).rolling(24).mean()
    df["delta"] = df["consumo_kwh"].diff().shift(1)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df = df.rename(columns={"timestamp": "ds", "consumo_kwh": "y"})
    df.dropna(inplace=True)

    return df

@task(name="sequential_validation")
def sequential_validation(df):
    rmse_values, mae_values = [], []
    predicciones_totales = pd.DataFrame()
    start_val = pd.Timestamp('2020-01-01')
    end_val = df['ds'].max().normalize()

    for dia in pd.date_range(start=start_val, end=end_val, freq='D'):
        train = df[df['ds'] < dia]
        test = df[(df['ds'] >= dia) & (df['ds'] < dia + pd.Timedelta(days=1))]
        if test.empty or len(train) < 48:
            continue

        df_festivos = pd.DataFrame({
            "holiday": "festivo",
            "ds": pd.to_datetime(FESTIVOS),
            "lower_window": 0,
            "upper_window": 1
        })

        m = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='additive',
            holidays=df_festivos
        )
        for r in regresores:
            m.add_regressor(r)
        m.fit(train[["ds", "y"] + regresores])
        forecast = m.predict(test[["ds"] + regresores])

        rmse = np.sqrt(mean_squared_error(test["y"], forecast["yhat"]))
        mae = mean_absolute_error(test["y"], forecast["yhat"])
        rmse_values.append(rmse)
        mae_values.append(mae)

        pred_dia = pd.DataFrame({
            "timestamp": test["ds"].values,
            "real": test["y"].values,
            "predicho": forecast["yhat"].values
        })
        predicciones_totales = pd.concat([predicciones_totales, pred_dia])

    return np.mean(rmse_values), np.mean(mae_values), predicciones_totales

@task(name="plot_results")
def plot_results(preds_total):
    plt.figure(figsize=(26, 8))
    plt.plot(preds_total["timestamp"], preds_total["real"], label="Real", alpha=0.6)
    plt.plot(preds_total["timestamp"], preds_total["predicho"], label="Predicción Prophet", linestyle="dashed")
    plt.xlabel("Fecha")
    plt.ylabel("Consumo (kWh)")
    plt.title("Comparación Consumo Real vs Predicción (Validación Temporal)")
    plt.legend()
    plt.grid(True)
    plot_path = "prophet_predictions.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

class ProphetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict(self, context, model_input):
        forecast = self.model.predict(model_input)
        return forecast[["ds", "yhat"]]

@task(name="register_model")
def register_model(df, rmse, mae, plot_path):
    fecha_validacion = pd.Timestamp("2020-01-01")
    df_train = df[df["ds"] < fecha_validacion].copy()

    df_festivos = pd.DataFrame({
        "holiday": "festivo",
        "ds": pd.to_datetime(FESTIVOS),
        "lower_window": 0,
        "upper_window": 1
    })

    modelo_final = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive',
        holidays=df_festivos
    )
    for r in regresores:
        modelo_final.add_regressor(r)
    modelo_final.fit(df_train[["ds", "y"] + regresores])

    mlflow.set_experiment("prophet_consumo_individual")
    with mlflow.start_run(run_name="prophet_consumo_individual") as run:
        run_id = run.info.run_id
        mlflow.log_metric("RMSE_validacion", rmse)
        mlflow.log_metric("MAE_validacion", mae)
        mlflow.log_artifact(plot_path)

        model_name = "Prophet-Consumo-Hogar"
        logged_model = mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=ProphetWrapper(modelo_final),
            conda_env=None
        )
        mlflow.register_model(model_uri=logged_model.model_uri, name=model_name)

@flow(name="prophet_consumo_flow", log_prints=True)
def prophet_consumo_flow(file_path: str = FILE_PATH_HOGAR):
    authenticate_dagshub()
    initialize_dagshub()
    df = load_and_prepare_data(file_path)
    rmse, mae, preds_total = sequential_validation(df)
    plot_path = plot_results(preds_total)
    register_model(df, rmse, mae, plot_path)
    print(f"RMSE Final: {rmse:.2f}, MAE Final: {mae:.2f}")

if __name__ == "__main__":
    prophet_consumo_flow()
