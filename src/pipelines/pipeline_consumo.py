import pandas as pd
import numpy as np
import os
from datetime import timedelta
from xgboost import XGBRegressor
import joblib
import mlflow.pyfunc
import dagshub
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Variables globales
modelo_consumo_path = 'modelo_consumo_reentrenado.pkl'
folder_output = 'datos_simulacion'
file_path_hogar = '../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv'

columnas_modelo_consumo = [
    "hour", "weekday", "is_weekend", "is_holiday",
    "tmed", "tmin", "tmax", "prec", "velmedia", "racha", "sol", "hrMedia", "año",
    "lag_1h", "lag_24h", "rolling_3h", "rolling_24h", "delta",
    "hour_sin", "hour_cos"
]


def inicializar_entorno_consumo():
    """
    Descripción:
        Inicializa el entorno del pipeline de consumo:
        - Conecta con DagsHub.
        - Elimina todos los archivos existentes en la carpeta de simulación.
        - Elimina el modelo reentrenado si ya existe.

    Params:
        Ninguno.

    Output:
        Ninguno (efectos secundarios: limpieza de entorno).
    """
    dagshub.init(repo_owner="auditoria.SGBA1", repo_name="Proyectos-SGBA1", mlflow=True)

    if os.path.exists(folder_output):
        for f in os.listdir(folder_output):
            os.remove(os.path.join(folder_output, f))
    else:
        os.makedirs(folder_output)

    if os.path.exists(modelo_consumo_path):
        os.remove(modelo_consumo_path)


def cargar_dataset_consumo():
    """
    Descripción:
        Carga el dataset original de consumo de la casa, incluyendo variables climáticas y estructurales.

    Params:
        Ninguno.

    Output:
        df (DataFrame): DataFrame con la columna 'timestamp' parseada como datetime.
    """
    return pd.read_csv(file_path_hogar, parse_dates=['timestamp'])


def preparar_features_consumo(df):
    """
    Descripción:
        Genera las variables necesarias para el modelo de predicción de consumo.
        Aplica ingeniería de features: temporales, lags, rolling, senos/cosenos horarios, etc.

    Params:
        df (DataFrame): DataFrame original con columna 'timestamp' y 'consumo_kwh'.

    Output:
        df (DataFrame): DataFrame con columnas de features completas y sin NaNs.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["año"] = df["timestamp"].dt.year
    df["lag_1h"] = df["consumo_kwh"].shift(1)
    df["lag_24h"] = df["consumo_kwh"].shift(24)
    df["rolling_3h"] = df["consumo_kwh"].rolling(3).mean()
    df["rolling_24h"] = df["consumo_kwh"].rolling(24).mean()
    df["delta"] = df["consumo_kwh"].diff()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df.dropna()


def cargar_modelo_consumo():
    """
    Descripción:
        Carga el modelo de MLflow desde el registry de DagsHub.

    Params:
        Ninguno.

    Output:
        modelo (mlflow.pyfunc.PyFuncModel): Modelo XGBoost en formato MLflow listo para predicción.
    """
    return mlflow.pyfunc.load_model("models:/XGBoost-Consumo-Hogar/latest")


def predecir_consumo_por_dia(dia, modelo):
    """
    Descripción:
        Genera predicciones de consumo horario para un día completo usando el modelo proporcionado.

    Params:
        dia (str o datetime): Día a predecir (por ejemplo "2020-01-01").
        modelo (MLflow o XGBoost): Modelo entrenado para hacer predicción.

    Output:
        DataFrame con columnas: fecha, pred_consumo, real_consumo, error.
    """
    df = cargar_dataset_consumo()
    fecha_inicio = pd.to_datetime(dia) - pd.Timedelta(hours=48)
    fecha_fin = pd.to_datetime(dia) + pd.Timedelta(hours=23)
    df_pred = df[(df["timestamp"] >= fecha_inicio) & (df["timestamp"] <= fecha_fin)].copy()
    df_pred = preparar_features_consumo(df_pred)
    df_dia = df_pred[df_pred["timestamp"].dt.date == pd.to_datetime(dia).date()].copy()

    resultados = []
    for _, row in df_dia.iterrows():
        X_row = row[columnas_modelo_consumo].values.reshape(1, -1)
        X_df = pd.DataFrame(X_row, columns=columnas_modelo_consumo)
        X_df = X_df.astype({
            "hour": int,
            "weekday": int,
            "is_weekend": int,
            "is_holiday": int,
            "tmed": float,
            "tmin": float,
            "tmax": float,
            "prec": float,
            "velmedia": float,
            "racha": float,
            "sol": float,
            "hrMedia": float,
            "año": int,
            "lag_1h": float,
            "lag_24h": float,
            "rolling_3h": float,
            "rolling_24h": float,
            "delta": float,
            "hour_sin": float,
            "hour_cos": float
        })

        y_real = row["consumo_kwh"]
        timestamp = row["timestamp"]
        y_pred = modelo.predict(X_df)[0]
        error = abs(y_real - y_pred)
        resultados.append({
            "fecha": timestamp,
            "pred_consumo": y_pred,
            "real_consumo": y_real,
            "error": error
        })

    return pd.DataFrame(resultados)


def guardar_resultados_consumo(df_resultado, dia):
    """
    Descripción:
        Guarda las predicciones horarias del día en un único archivo acumulado.
        Elimina las filas duplicadas si el día ya fue procesado.

    Params:
        df_resultado (DataFrame): DataFrame con columnas 'fecha', 'pred_consumo', 'real_consumo', 'error'.
        dia (str o datetime): Día correspondiente a los datos.

    Output:
        Archivo CSV en carpeta 'datos_simulacion' con nombre 'predicciones_consumo.csv'.
    """
    archivo_predicciones = f"{folder_output}/predicciones_consumo.csv"

    # Si ya existe, lo cargamos y eliminamos posibles filas duplicadas del mismo día
    if os.path.exists(archivo_predicciones):
        df_existente = pd.read_csv(archivo_predicciones, parse_dates=["fecha"])
        fecha_dia = pd.to_datetime(dia).date()
        df_existente = df_existente[df_existente["fecha"].dt.date != fecha_dia]
        df_completo = pd.concat([df_existente, df_resultado], ignore_index=True)
    else:
        df_completo = df_resultado

    df_completo.sort_values("fecha", inplace=True)
    df_completo.to_csv(archivo_predicciones, index=False)



def reentrenar_modelo_consumo(dia):
    """
    Descripción:
        Reentrena el modelo de consumo utilizando todos los datos hasta el día indicado.
        Guarda el nuevo modelo entrenado localmente como archivo .pkl.

    Params:
        dia (str o datetime): Día límite (no inclusive) para los datos de entrenamiento.

    Output:
        modelo (XGBRegressor): Modelo reentrenado.
    """
    df = cargar_dataset_consumo()
    df_hasta = df[df["timestamp"] < (pd.to_datetime(dia) + timedelta(days=1))].copy()
    df_train = preparar_features_consumo(df_hasta)
    X = df_train[columnas_modelo_consumo]
    y = df_train["consumo_kwh"]
    modelo = XGBRegressor()
    modelo.fit(X, y)
    joblib.dump(modelo, modelo_consumo_path)
    return modelo

def calcular_metricas_consumo(dia):
    """
    Descripción:
        Calcula las métricas MAE, RMSE y MAPE para un día a partir del archivo acumulado de predicciones.
        Las guarda en 'metricas_consumo.csv' (una fila por día, acumulativa).

    Params:
        dia (str o datetime): Día para el cual se evalúan las predicciones.

    Output:
        Imprime las métricas por consola y actualiza el archivo global de métricas.
    """
    archivo_pred = f"{folder_output}/predicciones_consumo.csv"
    archivo_metricas = f"{folder_output}/metricas_consumo.csv"
    dia_ts = pd.to_datetime(dia).normalize()

    if not os.path.exists(archivo_pred):
        print(f"No se encontró el archivo de predicciones acumulado.")
        return

    # Cargar todas las predicciones y filtrar por día
    df = pd.read_csv(archivo_pred, parse_dates=["fecha"])
    df_dia = df[df["fecha"].dt.normalize() == dia_ts]

    if df_dia.empty:
        print(f"No hay predicciones para el día {dia_ts.date()}.")
        return

    # Cálculo de métricas
    mae = mean_absolute_error(df_dia["real_consumo"], df_dia["pred_consumo"])
    rmse = root_mean_squared_error(df_dia["real_consumo"], df_dia["pred_consumo"])
    df_no_ceros = df_dia[df_dia["real_consumo"] != 0]
    mape = (abs((df_no_ceros["real_consumo"] - df_no_ceros["pred_consumo"]) / df_no_ceros["real_consumo"])).mean() * 100

    # Imprimir métricas
    print(f"MAE:  {mae:.5f} kWh")
    print(f"RMSE: {rmse:.5f} kWh")
    print(f"MAPE: {mape:.2f} %")

    # Crear nueva fila
    nueva_fila = pd.DataFrame([{
        "fecha": dia_ts,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }])

    # Actualizar archivo global de métricas
    if os.path.exists(archivo_metricas):
        df_metricas = pd.read_csv(archivo_metricas, parse_dates=["fecha"])
        df_metricas = df_metricas[df_metricas["fecha"] != dia_ts]
        df_metricas = pd.concat([df_metricas, nueva_fila], ignore_index=True)
    else:
        df_metricas = nueva_fila

    df_metricas.sort_values("fecha", inplace=True)
    df_metricas.to_csv(archivo_metricas, index=False)
