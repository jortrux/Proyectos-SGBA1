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
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.logging import get_run_logger
from datetime import timedelta

# Variables de entorno
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
REPO_DATA_DIR_PATH = os.getenv('REPO_DATA_DIR_PATH')

FILE_PATH_HOGAR = (
    REPO_DATA_DIR_PATH + "/data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
    if REPO_DATA_DIR_PATH
    else "../../../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
)

@task(name="autenticar_dagshub", retries=2)
def authenticate_dagshub():
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

@task(name="inicializar_dagshub")
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

@task(name="cargar_y_preparar_datos", cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    festivos = [
        "2017-01-06", "2017-04-14", "2017-04-17", "2017-05-01", "2017-06-05", "2017-06-24",
        "2017-08-15", "2017-09-11", "2017-09-25", "2017-10-12", "2017-11-01", "2017-12-06",
        "2017-12-08", "2017-12-25", "2017-12-26", "2018-01-01", "2018-01-06", "2018-03-30",
        "2018-04-02", "2018-05-01", "2018-08-15", "2018-09-11", "2018-10-12", "2018-11-01",
        "2018-12-06", "2018-12-08", "2018-12-25", "2018-12-26", "2019-01-01", "2019-04-19",
        "2019-04-22", "2019-05-01", "2019-06-10", "2019-06-24", "2019-08-15", "2019-09-11",
        "2019-09-24", "2019-10-12", "2019-11-01", "2019-12-06", "2019-12-25", "2019-12-26",
        "2020-01-01", "2020-01-06", "2020-04-10", "2020-04-13", "2020-05-01", "2020-06-01",
        "2020-06-24", "2020-08-15", "2020-09-11", "2020-09-24", "2020-10-12", "2020-12-08",
        "2020-12-25", "2020-12-26"
    ]
    df['is_holiday'] = df['timestamp'].dt.strftime('%Y-%m-%d').isin(festivos).astype(int)

    # Nuevas features
    df["lag_1h"] = df["consumo_kwh"].shift(1)
    df["lag_24h"] = df["consumo_kwh"].shift(24)
    df["rolling_3h"] = df["consumo_kwh"].rolling(3).mean()
    df["rolling_24h"] = df["consumo_kwh"].rolling(24).mean()
    df["delta"] = df["consumo_kwh"].diff()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df.dropna(inplace=True)

    return df

@task(name="dividir_train_test")
def split_data(df):
    train = df[df['timestamp'] < '2020-01-01']
    test = df[df['timestamp'] >= '2020-01-01']
    return train, test

@task(name="entrenar_modelo_xgboost")
def train_model(train):
    features = [
        'hour', 'weekday', 'is_weekend', 'is_holiday',
        'tmed', 'tmin', 'tmax', 'prec', 'velmedia', 'racha', 'sol', 'hrMedia', 'year',
        'lag_1h', 'lag_24h', 'rolling_3h', 'rolling_24h', 'delta',
        'hour_sin', 'hour_cos'
    ]
    target = 'consumo_kwh'
    umbral_min = train[train[target] > 0][target].min()

    def objective(trial):
        params = {
            'objective': 'reg:absoluteerror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        split_point = train['timestamp'].quantile(0.8)
        sub_train = train[train['timestamp'] < split_point]
        val = train[train['timestamp'] >= split_point]

        model = xgb.XGBRegressor(**params)
        model.fit(sub_train[features], sub_train[target])
        y_pred = model.predict(val[features])
        y_pred = np.clip(y_pred, umbral_min, None)
        return mean_absolute_error(val[target], y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(train[features], train[target])

    return best_model, study.best_params

@task(name="evaluar_modelo")
def evaluate_model(model, train, test):
    features = [
        'hour', 'weekday', 'is_weekend', 'is_holiday',
        'tmed', 'tmin', 'tmax', 'prec', 'velmedia', 'racha', 'sol', 'hrMedia', 'year',
        'lag_1h', 'lag_24h', 'rolling_3h', 'rolling_24h', 'delta',
        'hour_sin', 'hour_cos'
    ]
    target = 'consumo_kwh'
    umbral_min = train[train[target] > 0][target].min()

    rmse_values, mae_values = [], []
    errores_previos, predicciones_totales = [], pd.DataFrame()

    for dia in pd.date_range(start='2020-01-01', end=test['timestamp'].max(), freq='D'):
        test_dia = test[test['timestamp'].dt.date == dia.date()]
        if test_dia.empty:
            continue

        X_test = test_dia[features]
        y_test = test_dia[target]
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, umbral_min, None)

        ajuste_error = np.mean(errores_previos[-10:]) if len(errores_previos) > 10 else 0
        y_pred_ajustado = np.clip(y_pred + ajuste_error, umbral_min, None)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_ajustado))
        mae = mean_absolute_error(y_test, y_pred_ajustado)

        rmse_values.append(rmse)
        mae_values.append(mae)
        errores_previos.extend((y_test - y_pred).values)

        df_pred = pd.DataFrame({
            'timestamp': test_dia['timestamp'].values,
            'real': y_test.values,
            'predicho': y_pred,
            'predicho_ajustado': y_pred_ajustado
        })

        predicciones_totales = pd.concat([predicciones_totales, df_pred])

    return np.mean(rmse_values), np.mean(mae_values), predicciones_totales

@task(name="graficar_resultados")
def plot_results(df):
    plt.figure(figsize=(26, 8))
    plt.plot(df['timestamp'], df['real'], label="Real", alpha=0.6)
    plt.plot(df['timestamp'], df['predicho_ajustado'], label="Predicción Ajustada", linestyle="dashed", alpha=0.6)
    plt.legend()
    plt.grid(True)
    plt.title("Predicción de Consumo Energético")
    plot_path = "xgboost_validacion_mejorada.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@task(name="registrar_modelo_mlflow")
def register_model(model, rmse, mae, plot_path, best_params):
    mlflow.set_experiment("Predicción Consumo Hogar con XGBoost")
    with mlflow.start_run(run_name="xgboost_consumo_mejorado") as run:
        run_id = run.info.run_id
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_artifact(plot_path)

        model_uri = f"runs:/{run_id}/xgboost_model"
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        mlflow.register_model(model_uri=model_uri, name="XGBoost-Consumo-Hogar")
        print(f"Modelo registrado. Run ID: {run_id}")

@flow(name="flujo_xgboost_consumo")
def xgboost_workflow(file_path: str = FILE_PATH_HOGAR):
    df = load_data(file_path)
    train, test = split_data(df)
    model, best_params = train_model(train)
    rmse, mae, results = evaluate_model(model, train, test)
    plot_path = plot_results(results)
    authenticate_dagshub()
    initialize_dagshub()
    register_model(model, rmse, mae, plot_path, best_params)

if __name__ == "__main__":
    xgboost_workflow()
