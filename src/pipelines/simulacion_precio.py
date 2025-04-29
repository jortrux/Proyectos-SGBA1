
from pipeline_precio import (
    inicializar_entorno_precio,
    cargar_dataset_precio,
    preparar_features_precio,
    cargar_modelo_precio,
    entrenar_modelo_simple_precio,
    entrenar_modelo_precio_optuna,
    reentrenar_modelo_precio,
    log_full_experiment,
    predecir_precio,
)


import pandas as pd
import time
from datetime import timedelta

# Configurar simulación
# dia_actual = pd.to_datetime("2020-01-01")
# dia_final = pd.to_datetime("2020-03-01")

dia_actual = pd.to_datetime("2019-06-01")
dia_final = pd.to_datetime("2020-01-01")

inicializar_entorno_precio()

model, scaler, results_df, best_params = entrenar_modelo_precio_optuna(
    dia_fin_train=dia_actual, 
    dia_fin_test=dia_final, 
    n_trials=50)

log_full_experiment(
    model=model,
    scaler=scaler,
    resultados_df=results_df,
    best_params=best_params,
    run_name="XGBoost_Optuna_estacion" + str(dia_actual) + "---" + str(dia_final)
)

df_pred = predecir_precio(dia="2020-01-02", horas=72)

print(df_pred)

import matplotlib.pyplot as plt

# Sort the DataFrame by timestamp before plotting
df_pred = df_pred.sort_values(by='timestamp')

plt.figure(figsize=(12, 6))
plt.plot(df_pred['timestamp'], df_pred['prediccion_€/kwh'], marker='o', label='Predicción de Precio')
plt.plot(df_pred['timestamp'], df_pred['€/kwh'], marker='x', label='Precio Real')
plt.xlabel('Fecha')
plt.ylabel('Precio de la Luz')
plt.title('Predicción vs Precio Real de la Luz')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()