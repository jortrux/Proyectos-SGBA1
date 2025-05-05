import os
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
import dagshub
import numpy as np
from sklearn.metrics import mean_absolute_error

# Inicializar DAGsHub
dagshub.init(repo_owner="auditoria.SGBA1", repo_name="Proyectos-SGBA1", mlflow=True)

# Columnas utilizadas en el modelo completo (entrenado con clima + lags + rolling + ciclos)
columnas_modelo_completo = [
    "hour", "weekday", "is_weekend", "is_holiday",
    "tmed", "tmin", "tmax", "prec", "velmedia", "racha", "sol", "hrMedia", "año",
    "lag_1h", "lag_24h", "rolling_3h", "rolling_24h", "delta",
    "hour_sin", "hour_cos"
]

# Lista de festivos
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

# Ruta al dataset con clima y consumo
file_path_hogar = '../../../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv'

# Intentar cargar el dataset
try:
    df_clima = pd.read_csv(file_path_hogar, parse_dates=["timestamp"])
    dataset_cargado = True
except Exception as e:
    print("No se ha encontrado el clima. No se puede predecir.")
    df_clima = None
    dataset_cargado = False

def predecir_dia(fecha):
    if not dataset_cargado:
        print("No se ha encontrado el clima. No se puede predecir.")
        return

    fecha_dt = pd.to_datetime(fecha)

    # Usar desde 2 días antes para calcular bien rolling y lags
    fecha_inicio = fecha_dt - pd.Timedelta(hours=48)
    fecha_fin = fecha_dt + pd.Timedelta(hours=23)

    df_pred = df_clima[(df_clima["timestamp"] >= fecha_inicio) & (df_clima["timestamp"] <= fecha_fin)].copy()

    if df_pred.empty:
        print("No se han encontrado datos de clima suficientes para este rango. No se puede predecir.")
        return

    # Features temporales
    df_pred["hour"] = df_pred["timestamp"].dt.hour
    df_pred["weekday"] = df_pred["timestamp"].dt.weekday
    df_pred["is_weekend"] = (df_pred["weekday"] >= 5).astype(int)
    df_pred["is_holiday"] = df_pred["timestamp"].dt.strftime("%Y-%m-%d").isin(FESTIVOS).astype(int)

    # Variables adicionales del modelo (sin fuga de información)
    df_pred["lag_1h"] = df_pred["consumo_kwh"].shift(1)
    df_pred["lag_24h"] = df_pred["consumo_kwh"].shift(24)
    df_pred["rolling_3h"] = df_pred["consumo_kwh"].shift(1).rolling(3).mean()
    df_pred["rolling_24h"] = df_pred["consumo_kwh"].shift(1).rolling(24).mean()
    df_pred["delta"] = df_pred["consumo_kwh"].diff().shift(1)

    df_pred["hour_sin"] = np.sin(2 * np.pi * df_pred["hour"] / 24)
    df_pred["hour_cos"] = np.cos(2 * np.pi * df_pred["hour"] / 24)

    # Eliminar NaNs generados por lags/rolling
    df_pred = df_pred.dropna()

    # Filtrar solo el día que se quiere predecir
    df_pred = df_pred[df_pred["timestamp"].dt.date == fecha_dt.date()]

    if df_pred.empty:
        print(f"No hay suficientes datos para predecir correctamente el día {fecha}.")
        return

    # Separar consumo real si existe
    y_real = df_pred["consumo_kwh"] if "consumo_kwh" in df_pred.columns else None

    # Cargar modelo
    model = mlflow.pyfunc.load_model("models:/XGBoost-Consumo-Hogar/latest")
    X_pred = df_pred[columnas_modelo_completo]
    df_pred["yhat"] = model.predict(X_pred)

    # Visualización y métricas
    if y_real is not None:
        mae = mean_absolute_error(y_real, df_pred["yhat"])
        print(f"\nMAE (Error Medio Absoluto) para {fecha}: {mae:.4f} kWh")
        print("\nPredicciones vs Real:")
        print(df_pred[["timestamp", "consumo_kwh", "yhat"]])

        plt.figure(figsize=(10, 5))
        plt.plot(df_pred["timestamp"], df_pred["yhat"], marker='o', linestyle='--', label="Predicción")
        plt.plot(df_pred["timestamp"], y_real, marker='x', linestyle='-', label="Real")
        plt.xlabel("Hora del Día")
        plt.ylabel("Consumo (kWh)")
        plt.title(f"Predicción vs Real para {fecha} | MAE: {mae:.4f}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(df_pred["timestamp"], df_pred["yhat"], marker='o', linestyle='dashed', label="Predicción")
        plt.xlabel("Hora del Día")
        plt.ylabel("Consumo (kWh)")
        plt.title(f"Predicción de Consumo para {fecha}")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Ejemplo de uso
predecir_dia("2020-01-01")