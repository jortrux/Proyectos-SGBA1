from pipeline_precio import (
    inicializar_entorno_precio,
    entrenar_modelo_precio_optuna,
    reentrenar_modelo_precio,
    log_full_experiment,
    predecir_precio,
    guardar_resultados_precio,
)

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Configuración de Fechas
# =========================

fecha_inicio_entrenamiento = pd.to_datetime("2019-06-01")
fecha_inicio_prediccion = pd.to_datetime("2020-01-01")
fecha_fin_prediccion = pd.to_datetime("2020-03-01")
ventana_horas = 48  # Tamaño de la ventana para el modelo (48 horas)
num_trials_optuna = 50  # Número de pruebas para Optuna

# Calcular el número de horas a predecir
horas_a_predecir = int((fecha_fin_prediccion - fecha_inicio_prediccion).total_seconds() // 3600)

# =========================
# Función de Graficado
# =========================

def graficar_predicciones_precio(df_pred, num_registros):
    """
    Grafica las predicciones de precio vs el precio real.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_pred['timestamp'], df_pred['prediccion_€/kwh'], marker='o', label='Predicción de Precio')
    plt.plot(df_pred['timestamp'], df_pred['€/kwh'], marker='x', label='Precio Real')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de la Luz')
    plt.title(f'Predicción vs Precio Real ({num_registros} horas)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =========================
# Pipeline de Simulación
# =========================

def main():
    # Paso 1: Inicializar entorno
    inicializar_entorno_precio()
    print("Paso 1: Entorno inicializado.")

    # Paso 2: Entrenar modelo desde cero con Optuna
    model, scaler, resultados_metricas_df, best_params = entrenar_modelo_precio_optuna(
        dia_fin_train=fecha_inicio_entrenamiento,
        dia_fin_test=fecha_inicio_prediccion,
        block_size=ventana_horas,
        n_trials=num_trials_optuna,
    )
    print("Paso 2: Modelo entrenado con Optuna.")

    # Paso 3: Log de experimentos inicial
    log_full_experiment(
        model=model,
        scaler=scaler,
        resultados_df=resultados_metricas_df,
        best_params=best_params,
        run_name=f"simulacion_modelo_precio_optuna_{fecha_inicio_entrenamiento}---{fecha_inicio_prediccion}",
    )
    print("Paso 3: Experimento logueado.")

    # Paso 4: Reentrenar el mejor modelo con nuevos datos
    model, scaler, best_params = reentrenar_modelo_precio(
        dia_fin_train=fecha_inicio_entrenamiento,
        block_size=ventana_horas,
        base_run_name=f"{fecha_inicio_entrenamiento}---{fecha_inicio_prediccion}",
    )
    print("Paso 4: Modelo reentrenado.")

    log_full_experiment(
        model=model,
        scaler=scaler,
        resultados_df=None,
        best_params=best_params,
        run_name="reentrenamiento_modelo_precio",
    )
    print("Paso 4: Experimento de reentrenamiento logueado.")

    # Paso 5: Predecir el precio de la luz
    df_pred = predecir_precio(dia=fecha_inicio_prediccion, horas=horas_a_predecir)
    df_pred = df_pred.sort_values(by='timestamp')
    print("Paso 5: Predicción realizada.")

    graficar_predicciones_precio(df_pred, num_registros=horas_a_predecir)

    # Paso 6: Guardar predicciones
    guardar_resultados_precio(df_pred, dia=fecha_inicio_prediccion, horas=horas_a_predecir)
    print("Paso 6: Predicciones guardadas.")

if __name__ == "__main__":
    main()