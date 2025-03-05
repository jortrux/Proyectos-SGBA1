import mlflow

# Configurar la URI de seguimiento de MLflow en Dagshub
mlflow.set_tracking_uri("https://dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Configurar autenticación (sustituye 'TU_USUARIO' y 'TU_TOKEN')
mlflow.set_tracking_uri(f"https://auditoria.SGBA1:993b92c1e06e9737e8653be41dc9193b518d8fa5@dagshub.com/auditoria.SGBA1/SGBA1-smartgrids.mlflow")

# Verificar conexión listando experimentos
print(mlflow.search_models())
