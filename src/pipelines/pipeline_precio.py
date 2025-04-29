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
from sklearn.ensemble import RandomForestRegressor
import mlflow
from mlflow.models.signature import infer_signature
import joblib
import os

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
        "€/kwh", 
        "timestamp",
        "fecha", 
        # "anio",
        # "mes",
        # "dia",
        "dia_semana",
        # "estacion",
        # "tmed", 
        # "velmedia", 
        # "sol", 
        # "prec_valor",
        ]
    
    for column in df.columns:
        if not any([particula in column for particula in lista_particulas_mantener]):
            df_reducido.drop(column, axis=1, inplace=True)

    df_reducido["hora"] = df_reducido["timestamp"].dt.hour
    df_reducido["es_fin_de_semana"] = df_reducido["dia_semana"].isin([5, 6]).astype(int)
    df_reducido["es_laborable"] = (~df_reducido["dia_semana"].isin([5, 6])).astype(int)

    df_reducido["hora_seno"] = np.sin(2 * np.pi * df_reducido["hora"] / 24)
    df_reducido["hora_coseno"] = np.cos(2 * np.pi * df_reducido["hora"] / 24)
    df_reducido.drop("hora", axis=1, inplace=True)

    # df_reducido["estacion_seno"] = np.sin(2 * np.pi * df_reducido["estacion"] / 4)
    # df_reducido["estacion_coseno"] = np.cos(2 * np.pi * df_reducido["estacion"] / 4)
    # df_reducido.drop("estacion", axis=1, inplace=True)


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
        Carga el modelo de predicción de precios y el scaler desde DagsHub usando MLflow.

    Params:
        Ninguno.

    Output:
        model (Modelo): Modelo de predicción de precios.
        scaler (StandardScaler): Scaler usado para entrenar el modelo.
    """
    model = mlflow.pyfunc.load_model('models:/xgboost_precio_luz/latest')
    # Descargar el scaler desde los artefactos del modelo registrado
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions("xgboost_precio_luz", stages=["None", "Staging", "Production"])
    if not latest:
        raise ValueError("No hay versiones registradas del modelo.")
    run_id = latest[0].run_id
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler/scaler.pkl")
    scaler = joblib.load(scaler_path)
    return model, scaler

def entrenar_modelo_simple_precio(dia_fin_train, dia_fin_test, block_size=48):
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

        maes = []
        for i in range(n_blocks):
            start, end = i*block_size, (i+1)*block_size
            xb = test_X_scaled[start:end]
            yb = test_y.iloc[start:end]
            preds = model.predict(xb)
            maes.append(mean_absolute_error(yb, preds))

        mean_mae = np.mean(maes)

        return mean_mae

    # 5) Ejecuta estudio Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=10)
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

def log_full_experiment(model, scaler, resultados_df, best_params, run_name, train_X=None):
    """
    Registra en MLflow:
      - hiperparámetros,
      - métricas agregadas,
      - resultados por bloque (CSV),
      - modelo y scaler,
      - y registra el modelo en el Model Registry.
    """
    import mlflow
    from mlflow.models.signature import infer_signature
    import joblib
    mlflow.set_experiment("xgboost_precio_luz")
    with mlflow.start_run(run_name=run_name) as run:
        # Log params and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse_mean", resultados_df["rmse"].mean())
        mlflow.log_metric("mae_mean", resultados_df["mae"].mean())
        mlflow.log_metric("r2_mean",  resultados_df["r2"].mean())
        # Log results CSV
        resultados_csv = "resultados_bloques.csv"
        resultados_df.to_csv(resultados_csv, index=False)
        mlflow.log_artifact(resultados_csv, artifact_path="metrics")
        # Log scaler
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
        # Log model with signature
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
        # Register model in Model Registry
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="xgboost_precio_luz"
        )
        # Clean up
        import os
        os.remove(resultados_csv)
        os.remove(scaler_path)

def reentrenar_modelo_precio(dia_fin_train, dia_fin_test, block_size=48):
    """
    Descripción:
        Reentrena el modelo de predicción usando los hiperparametros de MLflow
        y evalúa su rendimiento en bloques deslizantes de 48h.

    Params:
        dia_fin_train (str): Fecha y hora de corte para el entrenamiento.
        dia_fin_test (str): Fecha y hora
    """

    experiment_name = "xgboost_precio_luz"
    # Obtener el experimento por nombre
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Buscar todas las runs y seleccionar la de menor MAE
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.mae_mean ASC"])
    if runs.empty:
        raise ValueError("No hay runs en el experimento para reentrenar el modelo.")

    best_run = runs.iloc[0]
    best_run_id = best_run.run_id

    # Cargar los mejores hiperparámetros de la run seleccionada
    best_params = mlflow.get_run(best_run_id).data.params

    # Convertir los parámetros a los tipos correctos
    for k in best_params:
        try:
            best_params[k] = int(best_params[k])
        except ValueError:
            try:
                best_params[k] = float(best_params[k])
            except ValueError:
                pass

    best_params['random_state'] = 42

    # Cargar y preparar datos
    df = cargar_dataset_precio()
    df_train, df_test = preparar_features_precio(df, dia_fin_train, dia_fin_test)

    train_X = df_train.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    train_X = train_X.select_dtypes(include=['int64', 'float64', 'bool'])
    train_y = df_train['€/kwh']

    # Descargar el scaler desde los artefactos de la mejor run
    scaler_artifact_path = mlflow.artifacts.download_artifacts(run_id=best_run_id, artifact_path="scaler/model.pkl")
    scaler = joblib.load(scaler_artifact_path)
    train_X_scaled = scaler.fit_transform(train_X)

    test_X = df_test.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    test_X = test_X.select_dtypes(include=['int64', 'float64', 'bool'])
    test_y = df_test['€/kwh']
    test_X_scaled = scaler.transform(test_X)

    # Reentrenar modelo
    model = XGBRegressor(**best_params)
    model.fit(train_X_scaled, train_y)

    # Evaluación por bloques
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
            
# def predecir_precio(dia, horas=48):
    """
    Realiza una predicción de precios para las próximas `horas` a partir de un día dado.
    
    Params:
        dia (str o datetime): Fecha y hora actual (inicio de predicción).
        horas (int): Cantidad de horas a predecir (por defecto 48).
    
    Output:
        df_resultados (DataFrame): DataFrame con las predicciones para las próximas `horas`.
    """
    import pandas as pd

    # Cargar el modelo desde MLflow Model Registry
    model, scaler = cargar_modelo_precio()

    # Cargar y preparar los datos
    df = cargar_dataset_precio()
    dia = pd.to_datetime(dia)
    dia_final = dia + pd.Timedelta(hours=horas-1)
    _, df_pred = preparar_features_precio(df, dia, dia_final)

    # Preparar X para predicción
    X_pred = df_pred.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    X_pred = X_pred.select_dtypes(include=['int64', 'float64', 'bool'])
    X_pred_scaled = scaler.transform(X_pred)

    # Realizar predicción
    y_pred = model.predict(X_pred_scaled)

    # Crear DataFrame de resultados
    df_resultados = df_pred.copy()
    df_resultados['prediccion_€/kwh'] = y_pred

    return df_resultados

def predecir_precio(dia, horas=48, model=None, scaler=None):
    import pandas as pd


    if model is None or scaler is None:
        model, scaler = cargar_modelo_precio()

    # Cargar y preparar los datos
    df = cargar_dataset_precio()
    dia = pd.to_datetime(dia)
    dia_final = dia + pd.Timedelta(hours=horas-1)
    _, df_pred = preparar_features_precio(df, dia, dia_final)

    # Preparar X para predicción
    X_pred = df_pred.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    X_pred = X_pred.select_dtypes(include=['int64', 'float64', 'bool'])
    X_pred_scaled = scaler.transform(X_pred)

    # Realizar predicción
    y_pred = model.predict(X_pred_scaled)

    df_resultados = df_pred.copy()
    df_resultados['prediccion_€/kwh'] = y_pred

    return df_resultados

def guardar_resultados_precio(df_resultados, dia):
    pass


def entrenar_modelo_rf_optuna(dia_fin_train, dia_fin_test, block_size=48, n_trials=50):
    """
    Entrena un RandomForestRegressor optimizando hiperparámetros con Optuna,
    evalúa en bloques de `block_size` horas y devuelve el modelo final,
    el scaler, el DataFrame de resultados por bloque y los mejores params.
    """
    df = cargar_dataset_precio()
    df_train, df_test = preparar_features_precio(df, dia_fin_train, dia_fin_test)

    train_X = df_train.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    train_X = train_X.select_dtypes(include=['int64', 'float64', 'bool'])
    train_y = df_train['€/kwh']

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)

    test_X = df_test.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    test_X = test_X.select_dtypes(include=['int64', 'float64', 'bool'])
    test_y = df_test['€/kwh']
    test_X_scaled = scaler.transform(test_X)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)
        model.fit(train_X_scaled, train_y)

        n_blocks = len(test_X_scaled) // block_size
        maes = []
        for i in range(n_blocks):
            start, end = i*block_size, (i+1)*block_size
            xb = test_X_scaled[start:end]
            yb = test_y.iloc[start:end]
            preds = model.predict(xb)
            maes.append(mean_absolute_error(yb, preds))
        return np.mean(maes)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    model = RandomForestRegressor(**best_params)
    model.fit(train_X_scaled, train_y)

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

def log_rf_experiment(model, scaler, resultados_df, best_params, run_name, train_X=None):
    """
    Registra en MLflow el experimento de RandomForest:
        - hiperparámetros,
        - métricas agregadas,
        - resultados por bloque (CSV),
        - modelo y scaler,
        - y registra el modelo en el Model Registry.
    """
    mlflow.set_experiment("randomforest_precio_luz")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse_mean", resultados_df["rmse"].mean())
        mlflow.log_metric("mae_mean", resultados_df["mae"].mean())
        mlflow.log_metric("r2_mean",  resultados_df["r2"].mean())
        resultados_csv = "resultados_bloques_rf.csv"
        resultados_df.to_csv(resultados_csv, index=False)
        mlflow.log_artifact(resultados_csv, artifact_path="metrics")
        scaler_path = "scaler_rf.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
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
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="randomforest_precio_luz"
        )
        os.remove(resultados_csv)
        os.remove(scaler_path)

def cargar_modelo_rf_precio():
    """
    Carga el modelo RandomForest y el scaler desde MLflow Model Registry.
    """
    model = mlflow.pyfunc.load_model('models:/randomforest_precio_luz/latest')
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions("randomforest_precio_luz", stages=["None", "Staging", "Production"])
    if not latest:
        raise ValueError("No hay versiones registradas del modelo RandomForest.")
    run_id = latest[0].run_id
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler/scaler_rf.pkl")
    scaler = joblib.load(scaler_path)
    return model, scaler

def predecir_precio_rf(dia, horas=48, model=None, scaler=None):
    """
    Realiza una predicción de precios usando RandomForest para las próximas `horas` a partir de un día dado.
    """

    if model is None or scaler is None:
        model, scaler = cargar_modelo_rf_precio()

    df = cargar_dataset_precio()
    dia = pd.to_datetime(dia)
    dia_final = dia + pd.Timedelta(hours=horas-1)
    _, df_pred = preparar_features_precio(df, dia, dia_final)

    X_pred = df_pred.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    X_pred = X_pred.select_dtypes(include=['int64', 'float64', 'bool'])
    X_pred_scaled = scaler.transform(X_pred)

    y_pred = model.predict(X_pred_scaled)
    df_resultados = df_pred.copy()
    df_resultados['prediccion_€/kwh'] = y_pred

    return df_resultados


if __name__ == "__main__":
    # Ejemplo de entrenamiento y logging con RandomForest + Optuna
    dia_fin_train = pd.to_datetime("2019-06-01")
    dia_fin_test = pd.to_datetime("2020-01-01")
    block_size = 48
    n_trials = 10  # Puedes aumentar para mejor búsqueda

    # Entrenamiento y optimización
    model_rf, scaler_rf, resultados_rf, best_params_rf = entrenar_modelo_rf_optuna(
        dia_fin_train, dia_fin_test, block_size=block_size, n_trials=n_trials
    )

    # Logging en MLflow
    df = cargar_dataset_precio()
    df_train, _ = preparar_features_precio(df, dia_fin_train, dia_fin_test)
    train_X_rf = df_train.drop(['€/kwh', 'timestamp', 'fecha'], axis=1, errors='ignore')
    train_X_rf = train_X_rf.select_dtypes(include=['int64', 'float64', 'bool'])
    log_rf_experiment(model_rf, scaler_rf, resultados_rf, best_params_rf, run_name="rf_optuna", train_X=train_X_rf)

    # Ejemplo de predicción con RandomForest
    dia_pred = "2020-3-2"
    horas_pred = 48
    df_pred_rf = predecir_precio_rf(dia_pred, horas=horas_pred, model=model_rf, scaler=scaler_rf)
    print(df_pred_rf.head())


