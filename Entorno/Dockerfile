FROM python:3.12-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir \
    dvc[gdrive] \
    prefect \
    dagshub \
    scikit-learn \
    pandas \
    numpy \
    xgboost \
    warnings

# Directorio de trabajo
WORKDIR /app

# Variables de entorno por defecto
ENV DAGSHUB_USERNAME='' \
    DAGSHUB_PASSWORD='' \
    GOOGLE_DRIVE_CREDENTIALS=''

# Puerto para Prefect UI
EXPOSE 8000

# Mantener el contenedor corriendo
CMD ["tail", "-f", "/dev/null"]
