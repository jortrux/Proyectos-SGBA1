import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
import dagshub

# Inicializar DAGsHub
dagshub.init(repo_owner='auditoria.SGBA1', repo_name='Proyectos-SGBA1', mlflow=True)

# Nombre del modelo en MLflow
model_name = "Prophet-Consumo-Hogar"

# Cargar el modelo desde MLflow Model Registry
model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

# Función para predecir un día específico
def predecir_dia(fecha):
    # Crear dataframe con todas las horas del día
    fecha_inicio = pd.to_datetime(fecha)
    horas = pd.date_range(start=fecha_inicio, periods=24, freq='H')
    df_pred = pd.DataFrame({'ds': horas})

    # Hacer la predicción
    forecast = model.predict(df_pred)

    # Mostrar resultados
    print(f"Predicciones para {fecha}:")
    print(forecast[['ds', 'yhat']])

    # Graficar la predicción
    plt.figure(figsize=(10, 5))
    plt.plot(forecast['ds'], forecast['yhat'], marker='o', linestyle='dashed', color='orange', label="Predicción")
    plt.xlabel("Hora del Día")
    plt.ylabel("Consumo (kWh)")
    plt.title(f"Predicción de Consumo para {fecha}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

# Ejemplo de uso: Cambia la fecha para predecir otro día
predecir_dia("2020-01-02")
