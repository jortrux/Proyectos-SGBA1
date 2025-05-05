
import pandas as pd
import numpy as np
import os
import joblib
from prophet import Prophet
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow.pyfunc
import dagshub
from prefect import task, flow, get_run_logger
import argparse


# Variables globales
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DOCKER_DATA_DIR = os.getenv("DOCKER_DATA_DIR")
KUBERNETES_PV_DIR = os.getenv("KUBERNETES_PV_DIR")

modelo_consumo_path = f'{KUBERNETES_PV_DIR}/modelo_consumo_reentrenado.pkl'
folder_output = f'{KUBERNETES_PV_DIR}/datos_simulacion_consumo'
file_path_hogar = f'{DOCKER_DATA_DIR}/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv'

FESTIVOS = [  # Festivos Barcelona
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

@task
def parse_arguments():
    parser = argparse.ArgumentParser(description='Reentrena y predice el consumo en un día determinado')
    parser.add_argument("--date", type=str, required=True, help="Fecha en formato YYYY-MM-DD")
    args = parser.parse_args()

    return pd.to_datetime(args.date)


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


@task
def inicializar_entorno_consumo(dia_actual):
    authenticate_dagshub()
    initialize_dagshub()

    if os.path.exists(folder_output):
        for f in os.listdir(folder_output):
            os.remove(os.path.join(folder_output, f))
    else:
        os.makedirs(folder_output)

    reentrenar_modelo_consumo(dia_actual)


@task
def cargar_dataset_consumo():
    return pd.read_csv(file_path_hogar, parse_dates=["timestamp"])


@task
def preparar_features_prophet(df):
    df = df.copy()
    df = df.rename(columns={"timestamp": "ds", "consumo_kwh": "y"})
    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day
    df["weekday"] = df["ds"].dt.weekday
    df["hour"] = df["ds"].dt.hour
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = df["ds"].dt.strftime("%Y-%m-%d").isin(FESTIVOS).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["lag_1h"] = df["y"].shift(1)
    df["lag_24h"] = df["y"].shift(24)
    df["rolling_3h"] = df["y"].shift(1).rolling(3).mean()
    df["rolling_24h"] = df["y"].shift(1).rolling(24).mean()
    df["delta"] = df["y"].diff().shift(1)
    df.dropna(inplace=True)
    return df


@task
def cargar_modelo_consumo():
    if not os.path.exists(modelo_consumo_path):
        raise FileNotFoundError("No se encontró el modelo local. Asegurate de ejecutar inicializar_entorno_consumo primero.")
    return joblib.load(modelo_consumo_path)


@task
def predecir_consumo_por_dia(dia, modelo):
    df = cargar_dataset_consumo()
    df = df.rename(columns={"timestamp": "ds", "consumo_kwh": "y"})
    fecha_inicio = pd.to_datetime(dia)
    df_historia = df[df["ds"] < fecha_inicio].copy()

    horas = pd.date_range(start=fecha_inicio, periods=24, freq='h')
    df_pred = pd.DataFrame({'ds': horas})
    df_pred["year"] = df_pred["ds"].dt.year
    df_pred["month"] = df_pred["ds"].dt.month
    df_pred["day"] = df_pred["ds"].dt.day
    df_pred["weekday"] = df_pred["ds"].dt.weekday
    df_pred["hour"] = df_pred["ds"].dt.hour
    df_pred["is_weekend"] = (df_pred["weekday"] >= 5).astype(int)
    df_pred["is_holiday"] = df_pred["ds"].dt.strftime("%Y-%m-%d").isin(FESTIVOS).astype(int)
    df_pred["hour_sin"] = np.sin(2 * np.pi * df_pred["hour"] / 24)
    df_pred["hour_cos"] = np.cos(2 * np.pi * df_pred["hour"] / 24)

    df_clima = df[["ds", 'tmed', 'tmin', 'tmax', 'prec', 'velmedia', 'racha', 'sol', 'hrMedia']].drop_duplicates('ds')
    df_pred = df_pred.merge(df_clima, on='ds', how='left')

    df_total = pd.concat([df_historia[["ds", "y"]], df_pred[["ds"]]], ignore_index=True)
    df_total = df_total.sort_values("ds").reset_index(drop=True)
    df_total["lag_1h"] = df_total["y"].shift(1)
    df_total["lag_24h"] = df_total["y"].shift(24)
    df_total["rolling_3h"] = df_total["y"].shift(1).rolling(3).mean()
    df_total["rolling_24h"] = df_total["y"].shift(1).rolling(24).mean()
    df_total["delta"] = df_total["y"].diff().shift(1)
    df_pred = df_pred.merge(df_total[["ds", "lag_1h", "lag_24h", "rolling_3h", "rolling_24h", "delta"]], on="ds", how="left")
    df_pred.fillna(0, inplace=True)

    forecast = modelo.predict(df_pred[["ds"] + regresores])
    df_real = df[(df["ds"] >= fecha_inicio) & (df["ds"] < fecha_inicio + pd.Timedelta(hours=24))][["ds", "y"]]
    resultado = forecast[["ds", "yhat"]].merge(df_real, on="ds", how="left")
    resultado.rename(columns={"ds": "fecha", "yhat": "pred_consumo", "y": "real_consumo"}, inplace=True)
    resultado["error"] = abs(resultado["pred_consumo"] - resultado["real_consumo"])
    return resultado


@task
def guardar_resultados_consumo(df_resultado, dia):
    archivo_pred = f"{folder_output}/predicciones_consumo.csv"
    if os.path.exists(archivo_pred):
        df_existente = pd.read_csv(archivo_pred, parse_dates=["fecha"])
        df_existente = df_existente[df_existente["fecha"].dt.date != pd.to_datetime(dia).date()]
        df_completo = pd.concat([df_existente, df_resultado], ignore_index=True)
    else:
        df_completo = df_resultado
    df_completo.sort_values("fecha", inplace=True)
    df_completo.to_csv(archivo_pred, index=False)


@task
def reentrenar_modelo_consumo(dia):
    df = cargar_dataset_consumo()
    df = df[df["timestamp"] < pd.to_datetime(dia)]
    df = preparar_features_prophet(df)
    
    from prophet import Prophet
    df_festivos = pd.DataFrame({
        "holiday": "festivo",
        "ds": pd.to_datetime(FESTIVOS),
        "lower_window": 0,
        "upper_window": 1
    })

    modelo = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='additive',
        holidays=df_festivos
    )
    for r in regresores:
        modelo.add_regressor(r)
    modelo.fit(df[["ds", "y"] + regresores])

    # Guardar en local
    joblib.dump(modelo, modelo_consumo_path)
    return modelo


@task
def calcular_metricas_consumo(dia):
    archivo_pred = f"{folder_output}/predicciones_consumo.csv"
    archivo_metricas = f"{folder_output}/metricas_consumo.csv"
    dia_ts = pd.to_datetime(dia).normalize()
    if not os.path.exists(archivo_pred):
        print("No se encontró el archivo de predicciones.")
        return

    df = pd.read_csv(archivo_pred, parse_dates=["fecha"])
    df_dia = df[df["fecha"].dt.normalize() == dia_ts]
    if df_dia.empty:
        print(f"No hay predicciones para {dia_ts.date()}")
        return

    mae = mean_absolute_error(df_dia["real_consumo"], df_dia["pred_consumo"])
    rmse = mean_squared_error(df_dia["real_consumo"], df_dia["pred_consumo"]) ** 0.5
    df_no_ceros = df_dia[df_dia["real_consumo"] != 0]
    mape = (abs((df_no_ceros["real_consumo"] - df_no_ceros["pred_consumo"]) / df_no_ceros["real_consumo"])).mean() * 100

    print(f"MAE: {mae:.5f} kWh | RMSE: {rmse:.5f} kWh | MAPE: {mape:.2f} %")
    nueva_fila = pd.DataFrame([{ "fecha": dia_ts, "MAE": mae, "RMSE": rmse, "MAPE": mape }])

    if os.path.exists(archivo_metricas):
        df_metricas = pd.read_csv(archivo_metricas, parse_dates=["fecha"])
        df_metricas = df_metricas[df_metricas["fecha"] != dia_ts]
        df_metricas = pd.concat([df_metricas, nueva_fila], ignore_index=True)
    else:
        df_metricas = nueva_fila

    df_metricas.sort_values("fecha", inplace=True)
    df_metricas.to_csv(archivo_metricas, index=False)


@flow
def flow_consumo():

    logger = get_run_logger()

    try:
        dia_actual = parse_arguments()

        inicializar_entorno_consumo(dia_actual)
        modelo = cargar_modelo_consumo()

        # 1. Predecir consumo
        df_resultados = predecir_consumo_por_dia(dia_actual, modelo)

        # 2. Imprimir por hora
        for _, row in df_resultados.iterrows():
            print(f"Fecha: {row['fecha']}")
            print(f"Predicción: {row['pred_consumo']:.3f} kWh")
            print(f"Real:       {row['real_consumo']:.3f} kWh")
            print(f"Error:      {row['error']:.3f} kWh\n")
            # time.sleep(1)  # Simulación temporal por hora

        # 3. Guardar CSV del día
        guardar_resultados_consumo(df_resultados, dia_actual)

        # 4. Calcular métricas e incluirlas en el CSV
        calcular_metricas_consumo(dia_actual)

        # 5. Reentrenar modelo con todos los datos hasta el día actual
        modelo = reentrenar_modelo_consumo(dia_actual)

        return 0
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(flow_consumo())
