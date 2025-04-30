from pipeline_precio import (
    inicializar_entorno_precio,
    entrenar_modelo_precio_optuna,
    reentrenar_modelo_precio,
    log_full_experiment,
    predecir_precio,
    guardar_resultados_precio,
)


import pandas as pd
import time
from datetime import timedelta
import matplotlib.pyplot as plt

def graficar_predicciones_precio(df_pred, registros):
    """
    Función para graficar las predicciones de precio.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(df_pred['timestamp'], df_pred['prediccion_€/kwh'], marker='o', label='Predicción de Precio')
    plt.plot(df_pred['timestamp'], df_pred['€/kwh'], marker='x', label='Precio Real')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de la Luz')
    plt.title(f'Predicción vs Precio Real ( {registros} horas)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Configurar simulación


dia_actual = pd.to_datetime("2019-06-01")
dia_presente = pd.to_datetime("2020-01-01")

dia_fin_registros = pd.to_datetime("2020-03-01")
numero_horas = int((dia_fin_registros - dia_presente).total_seconds() // 3600)

# paso 1: Inicializar entorno
inicializar_entorno_precio()
print("paso 1: Inicializar entorno")
# paso 2: Entrenar un modelo de 0 con optuna
model, scaler, resultados_metricas_df, best_params = entrenar_modelo_precio_optuna(
    dia_fin_train=dia_actual,
    dia_fin_test=dia_presente,
    block_size=48, # esto representa el tamaño de la ventana, 48 horas en este caso
    n_trials=10,

)
print("paso 2: Entrenar un modelo de 0 con optuna")

# Paso 3: Log de experimentos
log_full_experiment(
    model=model,
    scaler=scaler,
    resultados_df=resultados_metricas_df,
    best_params=best_params,
    run_name="simulacion_modelo_precio_optuna",
)
print("paso 3: Log de experimentos")
# Paso 4: Reentrenar el mejor modelo con nuevos datos
model, scaler, best_params = reentrenar_modelo_precio(
    dia_fin_train=dia_actual,
    block_size=48,
    run_name="2019-06-01 00:00:00---2020-01-01 00:00:00",
)
print("paso 4: Reentrenar el mejor modelo con nuevos datos")
log_full_experiment(
    model=model,
    scaler=scaler,
    resultados_df=None,
    best_params=best_params,
    run_name="reentrenamiento_modelo_precio",
)
print("paso 4: Log de experimentos")

# Paso 5 : Predecir el precio de la luz
df_pred = predecir_precio(dia=dia_presente, horas=numero_horas)
df_pred = df_pred.sort_values(by='timestamp')
print("paso 5: Predecir el precio de la luz")
graficar_predicciones_precio(df_pred, registros=numero_horas)
# Guardar predicciones

guardar_resultados_precio(df_pred, dia=dia_presente, horas=numero_horas)
print("paso 6: Guardar predicciones")