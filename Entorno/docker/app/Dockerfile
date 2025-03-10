FROM yagoutad/sgba1-base-image:latest

WORKDIR /opt/prefect

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar Prefect y dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los datos
COPY data/ /opt/prefect/data/

# Copiar el script del flujo
COPY flow_script.py .

# Ejecutar el flujo cuando se inicie el contenedor
CMD ["python", "flow_script.py"]
