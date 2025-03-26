import os
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
import dagshub
import numpy as np

# Inicializar DAGsHub
dagshub.init(repo_owner="auditoria.SGBA1", repo_name="Proyectos-SGBA1", mlflow=True)

# Nombre del modelo en MLflow
model_name = "XGBoost-Consumo-Hogar"

# Cargar el modelo desde MLflow Model Registry
model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

# Lista opcional de festivos (ejemplo) para la columna is_holiday
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

def predecir_dia(fecha):
    # Generar 24 horas para la fecha dada
    fecha_inicio = pd.to_datetime(fecha)
    horas = pd.date_range(start=fecha_inicio, periods=24, freq='H')
    df_pred = pd.DataFrame({"timestamp": horas})

    # Crear columnas de características
    df_pred["hour"] = df_pred["timestamp"].dt.hour
    df_pred["weekday"] = df_pred["timestamp"].dt.weekday
    df_pred["is_weekend"] = (df_pred["weekday"] >= 5).astype(int)
    df_pred["is_holiday"] = df_pred["timestamp"].dt.strftime("%Y-%m-%d").isin(FESTIVOS).astype(int)

    # Hacer la predicción usando las mismas columnas que usó el modelo
    # (hour, weekday, is_weekend, is_holiday) en el entrenamiento
    X_pred = df_pred[["hour", "weekday", "is_weekend", "is_holiday"]]
    df_pred["yhat"] = model.predict(X_pred)

    # Mostrar resultados
    print(f"Predicciones para {fecha}:")
    print(df_pred[["timestamp", "yhat"]])

    # Graficar la predicción
    plt.figure(figsize=(10, 5))
    plt.plot(df_pred["timestamp"], df_pred["yhat"], marker='o', linestyle='dashed', label="Predicción")
    plt.xlabel("Hora del Día")
    plt.ylabel("Consumo (kWh)")
    plt.title(f"Predicción de Consumo para {fecha}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

# Ejemplo de uso
predecir_dia("2020-01-02")
