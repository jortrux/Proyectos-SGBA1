import os
import subprocess
import pandas as pd
import numpy as np
import mlflow
import dagshub
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.logging import get_run_logger
from datetime import timedelta

# Variables de entorno
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
REPO_DATA_DIR_PATH = os.getenv('REPO_DATA_DIR_PATH')

# Ruta de archivo por defecto
FILE_PATH_HOGAR = (
    REPO_DATA_DIR_PATH + "/data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
    if REPO_DATA_DIR_PATH
    else "../../../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
)

def run_command(command: str) -> str:
    """Ejecuta un comando en la terminal."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error ejecutando '{command}': {e.stderr}")

@task(name="authenticate_dagshub", retries=2)
def authenticate_dagshub():
    """Autenticación con Dagshub."""
    logger = get_run_logger()
    try:
        token = os.getenv('DAGSHUB_TOKEN')
        if not token:
            raise ValueError("DAGSHUB_TOKEN no está configurado.")
        dagshub.auth.add_app_token(token=token)
        logger.info("Autenticación exitosa en Dagshub.")
        return True
    except Exception as e:
        logger.error(f"Fallo autenticando Dagshub: {e}")
        raise

@task(name="initialize_dagshub")
def initialize_dagshub():
    """Inicializa repositorio Dagshub."""
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
    """Carga y prepara los datos."""
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='ds')

    # Crear columnas simples
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['weekday'] = df['ds'].dt.weekday
    df['hour'] = df['ds'].dt.hour
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    festivos = [
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
    df['is_holiday'] = df['ds'].dt.strftime("%Y-%m-%d").isin(festivos).astype(int)

    # Normalizar consumo
    scaler = MinMaxScaler()
    df['consumo_kwh'] = scaler.fit_transform(df[['consumo_kwh']])
    return df, scaler

@task
def split_data(df):
    """Divide datos en entrenamiento y prueba, últimos 7 días como test."""
    split_date = df['ds'].max() - pd.Timedelta(days=7)
    train = df[df['ds'] <= split_date]
    test = df[df['ds'] > split_date]
    return train, test

@task
def train_xgboost_model(train, scaler):
    """Optimiza y entrena XGBoost."""
    features = ['hour','weekday','is_weekend','is_holiday']
    target = 'consumo_kwh'
    X_train, y_train = train[features], train[target]

    # Definimos búsqueda con Optuna
    def objective(trial):
        params = {
            "objective": "reg:pseudohubererror",
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return mean_squared_error(y_train, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # Entrenar con los mejores params
    best_params = study.best_params
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)

    return final_model, best_params

@task
def evaluate_model(model, train, test, scaler):
    """Evalúa el modelo en test."""
    features = ['hour','weekday','is_weekend','is_holiday']
    target = 'consumo_kwh'
    X_test, y_test = test[features], test[target]

    y_pred = model.predict(X_test)
    # Ajustamos para evitar valores negativos
    min_val = train[train["consumo_kwh"] > 0]["consumo_kwh"].min()
    y_pred = np.clip(y_pred, min_val, y_pred.max())

    # Desnormalizamos para métricas
    y_pred_descaled = scaler.inverse_transform(y_pred.reshape(-1,1))
    y_test_descaled = scaler.inverse_transform(y_test.values.reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(y_test_descaled, y_pred_descaled))
    mae = mean_absolute_error(y_test_descaled, y_pred_descaled)

    # Creamos un DF para graficar
    results_df = pd.DataFrame({
        "ds": test["ds"],
        "real": y_test_descaled.flatten(),
        "pred": y_pred_descaled.flatten()
    })
    return rmse, mae, results_df

@task
def plot_results(results_df):
    """Grafica resultados."""
    plt.figure(figsize=(10,5))
    plt.plot(results_df["ds"], results_df["real"], label="Real", marker='o')
    plt.plot(results_df["ds"], results_df["pred"], label="Predicción", marker='x')
    plt.title("Validación XGBoost")
    plt.xlabel("Fecha")
    plt.ylabel("Consumo kWh")
    plt.legend()
    plot_path = "xgboost_validation.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@task
def register_model_mlflow(model, rmse, mae, plot_path, best_params):
    """Registra modelo en MLflow."""
    mlflow.set_experiment("Predicción Consumo Hogar con XGBoost")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        for k,v in best_params.items():
            mlflow.log_param(k,v)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_artifact(plot_path)

        # Guardado del modelo
        model_name = "XGBoost-Consumo-Hogar"
        model_uri = f"runs:/{run_id}/model_xgb"
        mlflow.xgboost.log_model(model, artifact_path="model_xgb")
        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Modelo registrado. Run ID: {run_id}")

@flow(log_prints=True)
def xgboost_workflow(file_path: str = FILE_PATH_HOGAR):
    """Flujo principal usando XGBoost."""
    # Carga y separación
    df, scaler = load_data(file_path)
    train, test = split_data(df)

    # Entrenamiento
    model, best_params = train_xgboost_model(train, scaler)

    # Evaluación
    rmse, mae, results_df = evaluate_model(model, train, test, scaler)

    # Graficar
    plot_path = plot_results(results_df)

    # Autenticación e inicialización
    authenticate_dagshub()
    initialize_dagshub()

    # Registro en MLflow
    register_model_mlflow(model, rmse, mae, plot_path, best_params)

if __name__ == "__main__":
    # Llamamos al flujo
    xgboost_workflow()
