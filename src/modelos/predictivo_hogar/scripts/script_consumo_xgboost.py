import os
import subprocess
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import mlflow
import dagshub
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.logging import get_run_logger
from datetime import timedelta

# ============================================================================
# Variables de entorno y configuración
# ============================================================================
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
REPO_DATA_DIR_PATH = os.getenv('REPO_DATA_DIR_PATH')

# Si quieres cambiar la ruta del CSV, puedes ajustar esta variable
# o definirla vía una variable de entorno.
FILE_PATH_HOGAR = (
    REPO_DATA_DIR_PATH + "/data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
    if REPO_DATA_DIR_PATH
    else "../../../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv"
)

# ============================================================================
# Funciones auxiliares
# ============================================================================
def run_command(command: str) -> str:
    """
    Ejecuta un comando en la terminal y maneja errores.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error ejecutando '{command}': {e.stderr}")

@task(name="authenticate_dagshub", retries=2)
def authenticate_dagshub():
    """
    Tarea para autenticar con DagsHub usando las variables de entorno.
    """
    logger = get_run_logger()
    try:
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        if not dagshub_token:
            raise ValueError("La variable de entorno DAGSHUB_TOKEN no está configurada.")
        
        dagshub.auth.add_app_token(token=dagshub_token)
        logger.info("Autenticación con Dagshub exitosa.")
        return True
    except Exception as e:
        logger.error(f"Fallo al autenticar con Dagshub: {str(e)}")
        raise

@task(name="initialize_dagshub")
def initialize_dagshub():
    """
    Tarea para inicializar la conexión con DagsHub.
    """
    logger = get_run_logger()
    try:
        if not all([DAGSHUB_USERNAME, DAGSHUB_REPO_NAME]):
            raise ValueError("Faltan las variables de entorno: DAGSHUB_USERNAME o DAGSHUB_REPO_NAME.")
        
        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True
        )
        logger.info(f"Inicialización de Dagshub exitosa: {DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}")
        return True
    except Exception as e:
        logger.error(f"Fallo al inicializar Dagshub: {str(e)}")
        raise

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_and_prepare_data(file_path):
    """
    Cargar y preparar los datos a partir del CSV, incluyendo:
      - Conversión de timestamp a datetime
      - Generación de variables temporales (hora, día, etc.)
      - Detección de festivos
      - Normalización del consumo
    """
    # Lectura del CSV
    df = pd.read_csv(file_path)
    
    # Convertir timestamp a datetime y ordenar
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    # Crear columnas temporales
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Lista de festivos (ejemplo para 2017-2020)
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

    # Normalizar el consumo
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['consumo_kwh'] = scaler.fit_transform(df[['consumo_kwh']])

    # Obtener el consumo mínimo real para evitar predicciones negativas o cercanas a cero
    umbral_min = df[df['consumo_kwh'] > 0]['consumo_kwh'].min()

    # Separar en train y test
    train = df[df['timestamp'] < '2020-01-01']
    test = df[df['timestamp'] >= '2020-01-01']

    return df, train, test, scaler, umbral_min

@task
def optimize_xgboost(train, test, umbral_min):
    """
    Optimiza el modelo de XGBoost con Optuna.
    Devuelve los hiperparámetros óptimos.
    """
    features = ['hour', 'weekday', 'is_weekend', 'is_holiday']
    target = 'consumo_kwh'

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    def objective(trial):
        params = {
            'objective': 'reg:pseudohubererror',  # Función de pérdida robusta
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

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evitar valores negativos
        y_pred = np.clip(y_pred, umbral_min, max(y_pred))

        return mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    return best_params

@task
def train_best_model(train, test, scaler, umbral_min, best_params):
    """
    Entrena un modelo XGBoost con los mejores hiperparámetros y 
    realiza validación secuencial día a día.
    Devuelve:
      - DataFrame con predicciones totales
      - RMSE promedio
      - MAE promedio
      - Modelo entrenado (XGBRegressor)
    """
    features = ['hour', 'weekday', 'is_weekend', 'is_holiday']
    target = 'consumo_kwh'

    X_train, y_train = train[features], train[target]

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Validación secuencial
    rmse_values = []
    mae_values = []
    predicciones_totales = pd.DataFrame()
    errores_previos = []

    # Generamos rango de días en test
    df_combined = pd.concat([train, test])
    end_date = df_combined['timestamp'].max()

    for dia in pd.date_range(start='2020-01-01', end=end_date, freq='D'):
        test_dia = df_combined[df_combined['timestamp'].dt.date == dia.date()]
        if test_dia.empty:
            continue

        X_test_dia = test_dia[features]
        y_pred_dia = best_model.predict(X_test_dia)

        # Clip para evitar valores negativos / nulos
        y_pred_dia = np.clip(y_pred_dia, umbral_min, max(y_pred_dia))

        # Desnormalizamos
        y_pred_dia = scaler.inverse_transform(y_pred_dia.reshape(-1, 1))

        # Ajuste de error previo
        if len(errores_previos) > 10:
            ajuste_error = np.mean(errores_previos[-10:])
        else:
            ajuste_error = 0

        y_pred_ajustado = np.clip(y_pred_dia + ajuste_error, umbral_min, max(y_pred_dia))

        # En la serie original, el consumo sigue normalizado. Para calcular error correctamente,
        # debemos normalizar y_pred_ajustado o denormalizar la realidad.
        # Aquí, el test_dia['consumo_kwh'] está normalizado,
        # mientras que y_pred_ajustado está desnormalizado. Vamos a desnormalizar la 'real'.
        real_desnormalizada = scaler.inverse_transform(test_dia['consumo_kwh'].values.reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(real_desnormalizada, y_pred_ajustado))
        mae = mean_absolute_error(real_desnormalizada, y_pred_ajustado)

        rmse_values.append(rmse)
        mae_values.append(mae)

        # Guardar errores previos (en escala desnormalizada)
        errores_previos.extend(real_desnormalizada.flatten() - y_pred_dia.flatten())

        # DataFrame con predicciones
        df_pred = pd.DataFrame({
            'timestamp': test_dia['timestamp'].values,
            'real': real_desnormalizada.flatten(),
            'predicho': y_pred_dia.flatten(),
            'predicho_ajustado': y_pred_ajustado.flatten()
        })

        predicciones_totales = pd.concat([predicciones_totales, df_pred])

    rmse_final = np.mean(rmse_values)
    mae_final = np.mean(mae_values)

    return predicciones_totales, rmse_final, mae_final, best_model

@task
def plot_predictions(predicciones_totales):
    """
    Crear y guardar un gráfico de la comparativa entre valor real y predicho.
    Devuelve la ruta del archivo de la figura.
    """
    plt.figure(figsize=(26, 8))
    plt.plot(predicciones_totales['timestamp'], predicciones_totales['real'], label="Real", alpha=0.6)
    plt.plot(
        predicciones_totales['timestamp'],
        predicciones_totales['predicho_ajustado'],
        label="Predicción Ajustada",
        linestyle="dashed",
        alpha=0.6
    )
    plt.xlabel("Fecha")
    plt.ylabel("Consumo (kWh)")
    plt.title("Comparación Consumo Real vs. Predicción XGBoost (Optimizado con Optuna)")
    plt.legend()
    plt.grid(True)

    plot_path = "consumo_vs_prediccion_xgboost.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@task
def log_and_register_model_xgboost(best_params, rmse_final, mae_final, best_model, plot_path):
    """
    Registra el modelo en MLflow y Dagshub, subiendo hiperparámetros, métricas y la figura.
    """
    # Configurar experimento
    mlflow.set_experiment("xgboost_consumo_individual_optuna")

    with mlflow.start_run(run_name="xgboost_consumo_individual_optuna") as run:
        run_id = run.info.run_id
        model_name = "XGBoost-Consumo-Hogar"
        model_uri = f"runs:/{run_id}/xgboost_model"

        # Registrar hiperparámetros
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Registrar métricas
        mlflow.log_metric("RMSE", rmse_final)
        mlflow.log_metric("MAE", mae_final)

        # Subir gráfico
        mlflow.log_artifact(plot_path)

        # Guardar y registrar modelo en MLflow
        mlflow.xgboost.log_model(best_model, artifact_path="xgboost_model")

        # Registrar modelo en el Model Registry
        mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"Modelo registrado en MLflow con run_id: {run_id}")

@flow(name="xgboost_consumo_flow", log_prints=True)
def xgboost_consumo_flow(file_path: str = FILE_PATH_HOGAR):
    """
    Flujo principal de Prefect para:
      1) Cargar y preparar datos
      2) Optimizar hiperparámetros de XGBoost con Optuna
      3) Entrenar modelo con los mejores hiperparámetros
      4) Validar secuencialmente y obtener métricas
      5) Graficar y subir resultados a Dagshub/MLflow
    """
    # 1) Autenticación e inicialización en Dagshub
    authenticate_dagshub()
    initialize_dagshub()

    # 2) Cargar y preparar datos
    df, train, test, scaler, umbral_min = load_and_prepare_data(file_path)

    # 3) Optimizar hiperparámetros con Optuna
    best_params = optimize_xgboost(train, test, umbral_min)
    print("Mejores Hiperparámetros:", best_params)

    # 4) Entrenar y validar el mejor modelo
    predicciones_totales, rmse_final, mae_final, best_model = train_best_model(
        train, test, scaler, umbral_min, best_params
    )
    print(f"RMSE Final: {rmse_final:.2f}")
    print(f"MAE Final: {mae_final:.2f}")

    # 5) Graficar resultados
    plot_path = plot_predictions(predicciones_totales)

    # 6) Registrar modelo en Dagshub/MLflow
    log_and_register_model_xgboost(best_params, rmse_final, mae_final, best_model, plot_path)

    print("¡Proceso completado con éxito!")

# ============================================================================
# Ejecución directa del script
# ============================================================================
if __name__ == "__main__":
    xgboost_consumo_flow()
