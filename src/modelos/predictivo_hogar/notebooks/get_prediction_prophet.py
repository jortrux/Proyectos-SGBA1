import mlflow.pyfunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dagshub

# Inicializar DAGsHub
dagshub.init(repo_owner='auditoria.SGBA1', repo_name='Proyectos-SGBA1', mlflow=True)

# Cargar modelo desde el Model Registry de MLflow
model_name = "Prophet-Consumo-Hogar"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

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

# Regresores usados en el entrenamiento
regresores = [
    'hour', 'weekday', 'is_weekend', 'is_holiday',
    'tmed', 'tmin', 'tmax', 'prec', 'velmedia', 'racha', 'sol', 'hrMedia', 'year',
    'lag_1h', 'lag_24h', 'rolling_3h', 'rolling_24h', 'delta',
    'hour_sin', 'hour_cos'
]

def predecir_dia(fecha, df_historia_completa):
    fecha_inicio = pd.to_datetime(fecha)
    horas = pd.date_range(start=fecha_inicio, periods=24, freq='H')
    df_pred = pd.DataFrame({'ds': horas})

    # Base de tiempo
    df_pred["year"] = df_pred["ds"].dt.year
    df_pred["month"] = df_pred["ds"].dt.month
    df_pred["day"] = df_pred["ds"].dt.day
    df_pred["weekday"] = df_pred["ds"].dt.weekday
    df_pred["hour"] = df_pred["ds"].dt.hour
    df_pred["is_weekend"] = (df_pred["weekday"] >= 5).astype(int)
    df_pred["is_holiday"] = df_pred["ds"].dt.strftime("%Y-%m-%d").isin(FESTIVOS).astype(int)
    df_pred["hour_sin"] = np.sin(2 * np.pi * df_pred["hour"] / 24)
    df_pred["hour_cos"] = np.cos(2 * np.pi * df_pred["hour"] / 24)

    # Extraer clima por hora
    columnas_clima = ['ds', 'tmed', 'tmin', 'tmax', 'prec', 'velmedia', 'racha', 'sol', 'hrMedia']
    df_clima = df_historia_completa[columnas_clima].drop_duplicates('ds')
    df_pred = df_pred.merge(df_clima, on='ds', how='left')

    # Para calcular lags, usamos historia previa a la fecha de predicción
    df_historia = df_historia_completa[df_historia_completa['ds'] < fecha_inicio].copy()

    # Unir con predicción
    df_total = pd.concat([df_historia[["ds", "y"]], df_pred[["ds"]]], ignore_index=True)
    df_total = df_total.sort_values("ds").reset_index(drop=True)

    # Crear lags y rolling
    df_total["lag_1h"] = df_total["y"].shift(1)
    df_total["lag_24h"] = df_total["y"].shift(24)
    df_total["rolling_3h"] = df_total["y"].shift(1).rolling(3).mean()
    df_total["rolling_24h"] = df_total["y"].shift(1).rolling(24).mean()
    df_total["delta"] = df_total["y"].diff().shift(1)

    # Merge con df_pred
    df_pred = df_pred.merge(df_total[["ds", "lag_1h", "lag_24h", "rolling_3h", "rolling_24h", "delta"]], on="ds", how="left")
    df_pred.fillna(0, inplace=True)

    # Predicción
    forecast = model.predict(df_pred[["ds"] + regresores])

    # Consumo real para el mismo día
    df_real = df_historia_completa[df_historia_completa["ds"].between(fecha_inicio, fecha_inicio + pd.Timedelta(hours=23))][["ds", "y"]]

    # Merge predicción + real
    resultado = forecast[["ds", "yhat"]].merge(df_real, on="ds", how="left")

    # Mostrar tabla
    print(f"Predicciones para {fecha}:")
    print(resultado)

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(resultado['ds'], resultado['yhat'], marker='o', linestyle='--', color='orange', label="Predicción")
    plt.plot(resultado['ds'], resultado['y'], marker='o', linestyle='-', color='blue', label="Real")
    plt.xlabel("Hora del Día")
    plt.ylabel("Consumo (kWh)")
    plt.title(f"Consumo Real vs Predicción - {fecha}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Cargar historia hasta 2020-01-01 (para calcular features y clima)
df_hogar = pd.read_csv('../../../../data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv', parse_dates=["timestamp"])
df_hogar = df_hogar.rename(columns={"timestamp": "ds", "consumo_kwh": "y"})
df_historia = df_hogar[df_hogar["ds"] <= "2020-01-01 23:00:00"]

# Ejecutar predicción
predecir_dia("2020-01-01", df_historia)