
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
from prefect import task, flow, get_run_logger
import argparse

# Variables globales
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DOCKER_DATA_DIR = os.getenv("DOCKER_DATA_DIR")
KUBERNETES_PV_DIR = os.getenv("KUBERNETES_PV_DIR")

modelo_precio_path = f'{KUBERNETES_PV_DIR}/modelo_precio_reentrenado.pkl'
folder_output = f'{KUBERNETES_PV_DIR}/datos_simulacion_precio'
# file_path_precio = '../../data/processed/datos_precio/clima_precio_merged_recortado.parquet'
file_path_precio = f'{DOCKER_DATA_DIR}/processed/datos_precio/precio_consumo_electrico_timestamp_media.csv'


@task
def parse_arguments():
    parser = argparse.ArgumentParser(description='Reentrena y predice el consumo en un día determinado')
    parser.add_argument("--date", type=str, required=True, help="Fecha en formato YYYY-MM-DD")
    args = parser.parse_args()

    return pd.to_datetime(args.date)


@task(name="authenticate_dagshub", retries=2)
def authenticate_dagshub():
    logger = get_run_logger()
    token = os.getenv("DAGSHUB_TOKEN")
    if not token:
        raise ValueError("Falta DAGSHUB_TOKEN.")
    dagshub.auth.add_app_token(token=token)
    logger.info("Autenticación con Dagshub OK.")


@task(name="initialize_dagshub")
def initialize_dagshub():
    logger = get_run_logger()
    if not all([DAGSHUB_USERNAME, DAGSHUB_REPO_NAME]):
        raise ValueError("Falta DAGSHUB_USERNAME o DAGSHUB_REPO_NAME.")
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    logger.info("Dagshub inicializado.")


@task
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
    authenticate_dagshub()
    initialize_dagshub()

    if os.path.exists(folder_output):
        for f in os.listdir(folder_output):
            os.remove(os.path.join(folder_output, f))
    else:
        os.makedirs(folder_output)

    if os.path.exists(modelo_precio_path):
        os.remove(modelo_precio_path)


@task
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
    return pd.read_csv(file_path_precio)


@task
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
        "dia_semana",
        # "estacion",
        "tm", 
        "velmedia", 
        "racha",
        "dir_"
        "sol", 
        "prec_valor",
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

    # Añadir medias móviles y estadísticas de ventana para la columna de precio
    periodo_dia = 24
    periodo_semana = 24 * 7
    periodo_mes = 24 * 30
    periodo_año = 24 * 365
    window_sizes = [3, 6, 12, periodo_dia, periodo_semana, periodo_mes, periodo_año]
    for w in window_sizes:
        df_reducido[f'€/kwh_ma_{w}'] = df_reducido['€/kwh'].rolling(window=w, min_periods=1).mean()
        df_reducido[f'€/kwh_std_{w}'] = df_reducido['€/kwh'].rolling(window=w, min_periods=1).std()
        df_reducido[f'€/kwh_min_{w}'] = df_reducido['€/kwh'].rolling(window=w, min_periods=1).min()
        df_reducido[f'€/kwh_max_{w}'] = df_reducido['€/kwh'].rolling(window=w, min_periods=1).max()
    # Puedes añadir más estadísticas si lo deseas, por ejemplo, mediana:
        df_reducido[f'€/kwh_median_{w}'] = df_reducido['€/kwh'].rolling(window=w, min_periods=1).median()
    # Si tienes otras columnas numéricas relevantes, puedes repetir el proceso para ellas.

    # df_reducido["estacion_seno"] = np.sin(2 * np.pi * df_reducido["estacion"] / 4)
    # df_reducido["estacion_coseno"] = np.cos(2 * np.pi * df_reducido["estacion"] / 4)
    # df_reducido.drop("estacion", axis=1, inplace=True)


    # print("Dia actual:", dia_actual)
    # print("Dia final:", dia_final)
    # Filtrar el DataFrame de entrenamiento hasta el día actual (excluido)
    df_entrenamiento = df_reducido[df_reducido['timestamp'] < pd.to_datetime(dia_actual)].copy()

    # Filtrar el DataFrame de validación desde el día actual hasta el día final (incluido)
    df_validacion = df_reducido[(df_reducido['timestamp'] >= dia_actual) & (df_reducido['timestamp'] <= dia_final)].copy()

    # print(f"Fecha mínima entrenamiento: {df_entrenamiento['fecha'].iloc[1]}, máxima: {df_entrenamiento['fecha'].max()}")
    # print(f"Fecha mínima validación: {df_validacion['fecha'].min()}, máxima: {df_validacion['fecha'].max()}")

    return df_entrenamiento, df_validacion


@task
def cargar_modelo_precio():
    """
    Descripción:
        Carga el modelo de predicción de precios y el scaler desde DagsHub usando MLflow.
        Solo selecciona modelos cuyo nombre contiene 'reentrenado'.

    Params:
        Ninguno.

    Output:
        model (Modelo): Modelo de predicción de precios.
        scaler (StandardScanameler): Scaler usado para entrenar el modelo.
    """
    client = mlflow.tracking.MlflowClient()
    # Buscar todas las versiones del modelo
    all_versions = client.search_model_versions("name='xgboost_precio_luz'")
    # Filtrar solo las versiones cuyo run_name contiene 'reentrenado'
    reentrenado_versions = []
    for v in all_versions:
        run_id = v.run_id
        run = mlflow.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", "")
        if "reentrenamiento" in run_name.lower():
            reentrenado_versions.append(v)
    if not reentrenado_versions:
        raise ValueError("No hay versiones registradas del modelo con 'reentrenado' en el nombre.")
    # Seleccionar la versión más reciente (mayor version number)
    best_version = max(reentrenado_versions, key=lambda v: int(v.version))
    model_uri = f"models:/xgboost_precio_luz/{best_version.version}"
    model = mlflow.pyfunc.load_model(model_uri)
    # Descargar el scaler desde los artefactos del modelo registrado
    scaler_path = mlflow.artifacts.download_artifacts(run_id=best_version.run_id, artifact_path="scaler/scaler.pkl")
    scaler = joblib.load(scaler_path)
    return model, scaler


@task
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


@task
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
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
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
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


@task
def log_full_experiment(model, scaler, resultados_df, best_params, run_name, registrar_modelo=True):
    """
    Registra en MLflow:
      - hiperparámetros,
      - métricas agregadas,
      - resultados por bloque (CSV),
      - modelo y scaler,
      - y registra el modelo en el Model Registry.
    """
    import os

    mlflow.set_experiment("xgboost_precio_luz")
    with mlflow.start_run(run_name=run_name) as run:
        # Log params and metrics
        mlflow.log_params(best_params)
        if resultados_df is not None and not resultados_df.empty:
            mlflow.log_metric("rmse_mean", resultados_df["rmse"].mean())
            mlflow.log_metric("mae_mean", resultados_df["mae"].mean())
            mlflow.log_metric("r2_mean",  resultados_df["r2"].mean())
            # Log results CSV
            resultados_csv = os.path.join(folder_output, "resultados_bloques.csv")
            resultados_df.to_csv(resultados_csv, index=False)
            mlflow.log_artifact(resultados_csv, artifact_path="metrics")
        
        
        # Log scaler
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")
        # Log model with signature
        input_example = None
        signature = None

        mlflow.sklearn.log_model(
            model, 
            "model", 
            input_example=input_example, 
            signature=signature
        )
        # Register model in Model Registry solo si se solicita
        if registrar_modelo:
            mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="xgboost_precio_luz"
            )


@task
def reentrenar_modelo_precio(dia_fin_train, base_run_name, block_size=48):
    """
    Descripción:
        Reentrena el modelo de predicción usando los hiperparámetros de MLflow
        con todos los datos hasta `dia_fin_train` (sin validación, solo entrenamiento).

    Params:
        dia_fin_train (str o datetime): Fecha y hora de corte para el entrenamiento.
        block_size (int): No se usa en este caso, pero se mantiene por compatibilidad.

    Output:
        model: Modelo reentrenado.
        scaler: Scaler ajustado.
        best_params: Hiperparámetros usados.
    """

    experiment_name = "xgboost_precio_luz"
    # Obtener el experimento por nombre
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Buscar todas las runs con un nombre específico y seleccionar la de menor MAE
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '%{base_run_name}%'",
        order_by=["metrics.mae_mean ASC"]
    )
    if runs.empty:
        raise ValueError(f"No hay runs en el experimento que contengan '{base_run_name}' en el nombre para reentrenar el modelo.")

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
    # Filtrar solo hasta dia_fin_train (excluido)
    df_train = df[df['timestamp'] < pd.to_datetime(dia_fin_train)].copy()

    # Preparar features
    # Usar la misma lógica de preparar_features_precio pero solo para entrenamiento
    df_train, _ = preparar_features_precio(df, dia_fin_train, dia_fin_train)  # El segundo valor no se usa

    # Elimina las columnas si existen
    columnas_a_eliminar = ['€/kwh', 'timestamp', 'fecha']
    train_X = df_train.drop(columns=[col for col in columnas_a_eliminar if col in df_train.columns], axis=1)
    train_X = train_X.select_dtypes(include=['int64', 'float64', 'bool'])
    train_y = df_train['€/kwh']

    # Descargar el scaler desde los artefactos de la mejor run
    scaler_artifact_path = mlflow.artifacts.download_artifacts(run_id=best_run_id, artifact_path="scaler/scaler.pkl")
    scaler = joblib.load(scaler_artifact_path)
    train_X_scaled = scaler.fit_transform(train_X)

    # Reentrenar modelo
    model = XGBRegressor(**best_params)
    model.fit(train_X_scaled, train_y)

    # No hay evaluación ni resultados_df porque no hay validación
    return model, scaler, best_params


@task
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
    columnas_a_eliminar = ['€/kwh', 'timestamp', 'fecha']
    X_pred = df_pred.drop(columns=[col for col in columnas_a_eliminar if col in df_pred.columns], axis=1)
    X_pred = X_pred.select_dtypes(include=['int64', 'float64', 'bool'])
    X_pred_scaled = scaler.transform(X_pred)

    # Realizar predicción
    y_pred = model.predict(X_pred_scaled)

    df_resultados = df_pred.copy()
    df_resultados['prediccion_€/kwh'] = y_pred

    return df_resultados


@task
def guardar_resultados_precio(df_resultados, dia, horas=48):
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    # Formato de fecha solo con año, mes y día
    fecha_str = pd.to_datetime(dia).strftime('%Y%m%d')
    nombre_archivo = f"prediccion_precio.csv"
    ruta_archivo = os.path.join(folder_output, nombre_archivo)
    df_resultados.to_csv(ruta_archivo, index=False)


@flow
def flow_precio():

    logger = get_run_logger()

    try:
        fecha_inicio_entrenamiento = pd.to_datetime("2019-06-01")
        fecha_inicio_prediccion = parse_arguments()
        ventana_horas = 48  # Tamaño de la ventana para el modelo (48 horas)
        num_trials_optuna = 50  # Número de pruebas para Optuna

        # Calcular el número de horas a predecir
        horas_a_predecir = 24

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

        # Paso 6: Guardar predicciones
        guardar_resultados_precio(df_pred, dia=fecha_inicio_prediccion, horas=horas_a_predecir)
        print("Paso 6: Predicciones guardadas.")

        return 0
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(flow_precio())
