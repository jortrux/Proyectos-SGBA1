# Pasos

# Inicializar entrorno
import os
import dagshub
import pandas as pd
import mlflow
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna
from xgboost import XGBRegressor

# Variables globales
modelo_precio_path = 'modelo_precio_reentrenado.pkl'
folder_output = 'datos_simulacion_precio'
# file_path_precio = '../../data/processed/datos_precio/clima_precio_merged_recortado.parquet'
file_path_precio = '../../data/processed/datos_precio/clima_precio_merged_recortado.parquet'


def inicializar_entorno_precio():
    """
    Descripción:
        Inicializa el entorno del pipeline de precio:
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

    if os.path.exists(modelo_precio_path):
        os.remove(modelo_precio_path)

def cargar_dataset_precio():
    """
    Descripción:
        Carga el dataset original de precios de la electricidad.

    Params:
        Ninguno.

    Output:
        df (DataFrame): DataFrame con la columna 'timestamp' parseada como datetime.
    """
    # Cambia esta ruta al archivo CSV que contiene los datos de precios
    return pd.read_parquet(file_path_precio)

def preparar_features_precio(df, dia_actual, dia_final):
    """
    Descripción:
        Prepara las características para el modelo de predicción de precios.

    Params:
        df (DataFrame): DataFrame con los datos originales.

    Output:
        df (DataFrame): DataFrame con las características preparadas.
    """

    df_reducido = df.copy()
    lista_particulas_mantener = [
        "tm", 
        "viento", 
        "sol", 
        "€/kwh", 
        "fecha", 
        "anio",
        "mes",
        "dia",
        "dia_semana",
        "estacion"]
    
    for column in df.columns:
        if not any([particula in column for particula in lista_particulas_mantener]):
            df_reducido.drop(column, axis=1, inplace=True)

    # print("Dia actual:", dia_actual)
    # print("Dia final:", dia_final)
    # Filtrar el DataFrame de entrenamiento hasta el día actual (excluido)
    df_entrenamiento = df_reducido[df_reducido['timestamp'] < pd.to_datetime(dia_actual)].copy()

    # Filtrar el DataFrame de validación desde el día actual hasta el día final (incluido)
    df_validacion = df_reducido[(df_reducido['fecha'] >= dia_actual) & (df_reducido['fecha'] <= dia_final)].copy()

    # print(f"Fecha mínima entrenamiento: {df_entrenamiento['fecha'].iloc[1]}, máxima: {df_entrenamiento['fecha'].max()}")
    # print(f"Fecha mínima validación: {df_validacion['fecha'].min()}, máxima: {df_validacion['fecha'].max()}")

    return df_entrenamiento, df_validacion

def cargar_modelo_precio():
    """
    Descripción:
        Carga el modelo de predicción de precios desde DagsHub usando MLflow.

    Params:
        Ninguno.

    Output:
        model (Modelo): Modelo de predicción de precios.
    """
    model = mlflow.pyfunc.load_model('models:/xgboost_precio_electricidad/latest')
    return model

def entrenar_modelo_precio(dia_fin_train, dia_fin_test, block_size=48):
    """
    Entrena un modelo XGBoost usando datos hasta `dia_fin_train`,
    y evalúa su rendimiento en bloques deslizantes de 48h hasta `dia_fin_test`.
    """
    # Cargar y preparar datos
    df = cargar_dataset_precio()
    df_train, df_test = preparar_features_precio(df, dia_fin_train, dia_fin_test)

    # --- Preparar datos de entrenamiento ---
    train_X = df_train.drop(['€/kwh'], axis=1, errors='ignore')
    train_X = train_X.select_dtypes(include=['int64', 'float64', 'bool'])
    train_y = df_train['€/kwh']

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)

    # --- Entrenar modelo ---
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(train_X_scaled, train_y)

    # --- Evaluación en bloques de 48h ---
    test_X = df_test.drop(['€/kwh'], axis=1, errors='ignore')
    test_X = test_X.select_dtypes(include=['int64', 'float64', 'bool'])
    test_y = df_test['€/kwh']

    if test_X.shape[0] == 0:
        print("No hay datos de prueba disponibles para la evaluación.")
        resultados_df = pd.DataFrame()
        return model, scaler, resultados_df

    test_X_scaled = scaler.transform(test_X)
    test_X_scaled = pd.DataFrame(test_X_scaled, columns=test_X.columns)

    n_blocks = len(test_X_scaled) // block_size
    resultados = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size

        X_block = test_X_scaled.iloc[start:end]
        y_block = test_y.iloc[start:end]

        y_pred = model.predict(X_block)

        rmse = root_mean_squared_error(y_block, y_pred)
        mae = mean_absolute_error(y_block, y_pred)
        r2 = r2_score(y_block, y_pred)

        resultados.append({
            'bloque': i + 1,
            'inicio_bloque': df_test.index[start],
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

    resultados_df = pd.DataFrame(resultados)
    return model, scaler, resultados_df

def entrenar_modelo_precio_optuna(dia_fin_train, dia_fin_test, block_size=48, n_trials=50):
    """
    Entrena un XGBRegressor optimizando hiperparámetros con Optuna,
    evalúa en bloques de `block_size` horas y devuelve el modelo final
    junto al scaler, el DataFrame de resultados por bloque y los mejores params.
    """
    # 1) Carga y prepara datos
    df = cargar_dataset_precio()
    df_train, df_test = preparar_features_precio(df, dia_fin_train, dia_fin_test)

    # 2) Prepara X/y de train
    train_X = df_train.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    train_X = train_X.select_dtypes(include=['int64', 'float64', 'bool'])
    train_y = df_train['€/kwh']

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)

    # 3) Prepara X/y de test para objetivo Optuna y para evaluación final
    test_X = df_test.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    test_X = test_X.select_dtypes(include=['int64', 'float64', 'bool'])
    test_y = df_test['€/kwh']
    test_X_scaled = scaler.transform(test_X)

    # 4) Función objetivo para Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42
        }
        model = XGBRegressor(**params)
        model.fit(train_X_scaled, train_y)
        
        # cálculo por bloques
        n_blocks = len(test_X_scaled) // block_size
        rmses = []
        for i in range(n_blocks):
            start, end = i*block_size, (i+1)*block_size
            xb = test_X_scaled[start:end]
            yb = test_y.iloc[start:end]
            preds = model.predict(xb)
            rmses.append(root_mean_squared_error(yb, preds))

        return np.mean(rmses)

    # 5) Ejecuta estudio Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['random_state'] = 42

    # 6) Reentrena modelo con mejores hiperparámetros
    model = XGBRegressor(**best_params)
    model.fit(train_X_scaled, train_y)

    # 7) Evaluación por bloques
    n_blocks = len(test_X_scaled) // block_size
    resultados = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        Xb = test_X_scaled[start:end]
        yb = test_y.iloc[start:end]
        y_pred = model.predict(Xb)
        resultados.append({
            'bloque': i+1,
            'inicio_bloque': df_test.index[start],
            'rmse': root_mean_squared_error(yb, y_pred),
            'mae': mean_absolute_error(yb, y_pred),
            'r2': r2_score(yb, y_pred)
        })

    resultados_df = pd.DataFrame(resultados)
    return model, scaler, resultados_df, best_params


def log_experiment(model, scaler, resultados_df, best_params, train_X=None):
    """
    Registra en MLflow:
      - hiperparámetros,
      - métricas agregadas,
      - resultados por bloque (CSV),
      - modelo y scaler.
    """
    import mlflow
    from mlflow.models.signature import infer_signature
    import joblib
    mlflow.set_experiment("xgboost_precio_electricidad")
    with mlflow.start_run(run_name="XGB_Optuna_best"):
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse_mean", resultados_df["rmse"].mean())
        mlflow.log_metric("mae_mean", resultados_df["mae"].mean())
        mlflow.log_metric("r2_mean",  resultados_df["r2"].mean())
        resultados_csv = "resultados_bloques.csv"
        resultados_df.to_csv(resultados_csv, index=False)
        mlflow.log_artifact(resultados_csv, artifact_path="metrics")
        # Prepare input_example and signature for model logging
        if train_X is not None:
            input_example = train_X.iloc[:2]
            signature = infer_signature(train_X, model.predict(train_X))
        else:
            input_example = None
            signature = None
        mlflow.sklearn.log_model(
            model, 
            "model", 
            input_example=input_example, 
            signature=signature
        )
        # Save scaler as artifact, not as model
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

def predecir_precio_48h(dia):
    """
    Descripción:
        Realiza una predicción de precios para las próximas 48 horas.

    Params:
        dia (datetime): Fecha y hora actual.
        modelo (Modelo): Modelo de predicción de precios.

    Output:
        df_resultados (DataFrame): DataFrame con las predicciones y resultados.
    """
    
    # Cargar el modelo
    model = cargar_modelo_precio()


    pass
    

def guardar_resultados_precio(df_resultados, dia):
    pass





    


