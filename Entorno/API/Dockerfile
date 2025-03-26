# Usamos una imagen base de Python
FROM python:3.10

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los archivos del proyecto al contenedor
COPY . /app

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Configurar credenciales de DAGsHub dentro del contenedor
ENV DAGSHUB_USERNAME="auditoria.SGBA1"
ENV DAGSHUB_TOKEN="ee9be1f2d99f10b3647e4bccee075e65178ecf03"

# Configurar autenticación para DAGsHub (crea el archivo .netrc)
RUN echo "machine dagshub.com\nlogin ${DAGSHUB_USERNAME}\npassword ${DAGSHUB_TOKEN}" > ~/.netrc
RUN chmod 600 ~/.netrc

# Exponer el puerto 8000 para FastAPI
EXPOSE 8000

# Comando para ejecutar FastAPI con Uvicorn
CMD ["uvicorn", "fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]
