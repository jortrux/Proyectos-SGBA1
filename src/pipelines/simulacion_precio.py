
from pipeline_precio import (
    inicializar_entorno_precio,
    cargar_dataset_precio,
    preparar_features_precio,
    cargar_modelo_precio,
    # predecir_precio_48h
    entrenar_modelo_precio,
    entrenar_modelo_precio_optuna,
    log_experiment,
    predecir_precio_48h,
)


import pandas as pd
import time
from datetime import timedelta

# Configurar simulación
# dia_actual = pd.to_datetime("2020-01-01")
# dia_final = pd.to_datetime("2020-03-01")

# Inicializar entorno y cargar modelo inicial desde MLflow
# Paso 1: Inicializar entorno
inicializar_entorno_precio()
# # Paso 2: Cargar dataset
# df_precio = cargar_dataset_precio()
# # Paso 3: Preparar features
# df_train, df_test = preparar_features_precio(df_precio, dia_actual, dia_final)
# print(df_train.tail())
# print(df_test.head())
# print(df_test.tail())
# # Paso 4: Cargar modelo
# modelo = cargar_modelo_precio()
# print(modelo)
# # Paso 5: Predecir precio
# # df_resultados = predecir_precio_48h(dia_actual, modelo)

# Paso 6: Entrenar modelo

# model, scaler, resultados_df = entrenar_modelo_precio(
#     dia_fin_train='2019-01-01',
#     dia_fin_test='2020-01-01',
# )

# Paso 7: Entrenar modelo con Optuna
# model, scaler, resultados_df, best_params = entrenar_modelo_precio_optuna(
#     dia_fin_train='2019-01-01',
#     dia_fin_test='2020-01-01',
# )

# log_experiment(
#     model=model,
#     scaler=scaler,
#     resultados_df=resultados_df,
#     best_params=best_params
# )

# Realizar predicciones
predecir_precio_48h("2020-01-01")