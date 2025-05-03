from pipeline_consumo import (
    inicializar_entorno_consumo,
    cargar_modelo_consumo,
    predecir_consumo_por_dia,
    guardar_resultados_consumo,
    reentrenar_modelo_consumo,
    calcular_metricas_consumo 
)

import pandas as pd
import time
from datetime import timedelta

# Configurar simulación
dia_actual = pd.to_datetime("2020-01-01")
dia_final = pd.to_datetime("2020-03-01")

# Inicializar entorno y cargar modelo inicial desde MLflow
inicializar_entorno_consumo()
modelo = cargar_modelo_consumo()

# Simulación diaria
while dia_actual < dia_final:
    print(f"\nSimulando día {dia_actual.date()}")

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

    # 6. Avanzar al siguiente día
    dia_actual += timedelta(days=1)
