# Instalación y Configuración de Entorno para Entrenamiento de Modelos en Kubernetes

Este documento detalla los pasos para instalar y configurar un entorno en el que poder entrenar modelos de aprendizaje automático en un clúster de Kubernetes en local. Se utilizarán tecnologías como Docker, Prefect, DagsHub y MLflow en un sistema Linux.

---

## 1. Instalación de Docker

Docker es una plataforma que permite crear, ejecutar y administrar contenedores. Es el primer paso para configurar nuestro entorno.

### Pasos

1. **Actualizar la lista de paquetes:**

    ```bash
    sudo apt-get update
    ```

    Esto asegura que los paquetes estén actualizados antes de la instalación.

2. **Instalar dependencias necesarias para APT sobre HTTPS:**

    ```bash
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    ```

    Estas herramientas permiten a `apt` descargar e instalar paquetes de manera segura.

3. **Agregar la clave GPG oficial de Docker:**

    ```bash
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    ```

    Docker firma sus paquetes con una clave GPG para garantizar su autenticidad.

4. **Agregar el repositorio oficial de Docker:**

    ```bash
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    ```

    Esto agrega el repositorio de Docker a la lista de fuentes de APT.

5. **Actualizar la lista de paquetes nuevamente:**

    ```bash
    sudo apt-get update
    ```

    Para que el sistema reconozca el nuevo repositorio.

6. **Instalar Docker:**

    ```bash
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    ```

    Se instalan el motor de Docker, su CLI y el runtime de contenedores.

7. **Agregar el usuario al grupo Docker:**

    ```bash
    sudo usermod -aG docker $USER
    ```

    Esto permite ejecutar Docker sin `sudo`. **Es necesario cerrar sesión y volver a iniciarla para que los cambios surtan efecto.**

8. **Verificar la instalación:**

    ```bash
    docker --version
    ```

    Si la instalación fue exitosa, este comando mostrará la versión de Docker instalada.

---

## 2. Instalación de kubectl y Minikube

`kubectl` es la herramienta de línea de comandos para gestionar clústeres de Kubernetes. `Minikube` permite ejecutar un clúster de Kubernetes en local.

### Pasos

1. **Descargar la clave de firma de Google Cloud:**

    ```bash
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/kubernetes-archive-keyring.gpg
    ```

2. **Agregar el repositorio de Kubernetes:**

    ```bash
    echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
    ```

3. **Actualizar la lista de paquetes:**

    ```bash
    sudo apt-get update
    ```

4. **Instalar `kubectl`:**

    ```bash
    sudo apt-get install -y kubectl
    ```

5. **Verificar la instalación de `kubectl`:**

    ```bash
    kubectl version --client
    ```

6. **Descargar el binario más reciente de Minikube:**

    ```bash
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    ```

7. **Instalar Minikube:**

    ```bash
    sudo install minikube-linux-amd64 /usr/local/bin/minikube
    ```

8. **Eliminar el archivo descargado:**

    ```bash
    rm minikube-linux-amd64
    ```

9. **Iniciar Minikube con Docker:**

    ```bash
    minikube start --driver=docker
    ```

10. **Verificar el estado de Minikube:**

    ```bash
    minikube status
    ```

11. **Obtener información del clúster de Kubernetes:**

    ```bash
    kubectl cluster-info
    ```

---

## 3. Configuración del Proyecto y Variables de Entorno

### Crear la estructura de directorios

```bash
mkdir ml-training-platform
cd ml-training-platform
mkdir -p {src,config,kubernetes,docker,data,models,notebooks,.env}
touch .env/.env
touch .gitignore
touch README.md
```

Esto organiza el proyecto en carpetas para el código fuente, configuraciones, manifiestos de Kubernetes, datos, modelos y notebooks.

### Configurar las variables de entorno

```bash
cat > .env/.env.template << 'EOF'
# DagsHub Configuration
DAGSHUB_USERNAME=your_username
DAGSHUB_TOKEN=your_token
DAGSHUB_REPO_NAME=your_repo_name

# Prefect Configuration
PREFECT_API_KEY="your_api_key"

# Data Paths
DATA_DIR=../data
MODELS_DIR=../models
EOF

cp .env/.env.template .env/.env
```

Este archivo almacena credenciales y rutas esenciales sin exponerlas directamente en el código. Tendrás que rellenar el archivo .env real con tus credenciales de DagsHub y Prefect:

### Configurar `.gitignore`

```bash
cat > .gitignore << 'EOF'
# Archivos de entorno
.env/.env

# Directorios de datos y modelos
data/*
models/*

# Archivos de Python
__pycache__/
*.py[cod]
*$py.class

# Archivos de Jupyter Notebook
.ipynb_checkpoints

# Configuración de IDE
.idea/
.vscode/
*.swp
*.swo

# Archivos del sistema
.DS_Store
EOF
```

Esto evita subir archivos sensibles o innecesarios a Git.

### Crear un script para cargar las variables de entorno en Python

```bash
cat > src/config.py << 'EOF'
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / '.env' / '.env')

DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')

DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'data'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', PROJECT_ROOT / 'models'))

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EOF
```

Este script permite acceder a las variables de entorno desde Python y garantiza que las carpetas de datos y modelos existan.

### Crear `requirements.txt` e instalar dependencias

```bash
cat > requirements.txt << 'EOF'
numpy>=2.2.0
pandas>=2.2.0
python-dotenv>=1.0.0
scikit-learn>=1.6.0
xgboost>=2.1.0
prefect>=3.2.0
mlflow>=2.20.0
dagshub>=0.5.0
kubernetes>=32.0.0
EOF

pip install -r requirements.txt
```

Esto instala las librerías necesarias para manejar variables de entorno (`dotenv`) y para el seguimiento de experimentos con `MLflow`.

### Crea un script de ejemplo para verificar que todo esté configurado correctamente

```bash
cat > src/test_setup.py << 'EOF'
import mlflow
from mlflow.entities.view_type import ViewType
import dagshub
import dagshub.auth
from config import DAGSHUB_USERNAME, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN

def test_mlflow_connection():
    # Authenticate
    dagshub.auth.add_app_token(token=DAGSHUB_TOKEN)

    # Setup DagsHub
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    # Try to list experiments
    try:
        experiments = mlflow.search_experiments(view_type=ViewType.ALL)
        print("Successfully connected to MLflow!")
        print(f"Found {len(experiments)} experiments")
        return True
    except Exception as e:
        print(f"Failed to connect to MLflow: {str(e)}")
        return False

if __name__ == "__main__":
    test_mlflow_connection()
EOF
```

### Ejecuta el script de ejemplo

```bash
python src/test_setup.py
```

---

## 4. Integración de Kubernetes en el flujo de trabajo

### Crear y subir la imagen de docker donde se entrenarán los modelos

Para ello primero necesitamos crear y subir la imágen de docker en la que entrenaremos los modelos

```bash
# Create the Dockerfile
cat > docker/Dockerfile << 'EOF'
# Start with the slim Python image
FROM python:3.12-slim

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create working directory
WORKDIR /app

# The entrypoint will be the python command
ENTRYPOINT ["python"]
EOF

# Copy the requirements.txt in to the docker directory
cp requirements.txt docker/
```

```bash
# Create a script to handle the build and push of the image
cat > docker/build_and_push.sh << 'EOF'
#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define image name and tag
# Note: Replace 'your-registry' with your actual registry (like your Docker Hub username)
IMAGE_NAME="your-registry/ml-training"
IMAGE_TAG="latest"

# Build the image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ${DIR}

# Push the image
echo "Pushing Docker image..."
docker push ${IMAGE_NAME}:${IMAGE_TAG}
EOF

chmod +x docker/build_and_push.sh
```

Cambia el nombre del registro inicia, sesión en Docker Hub con tu cuenta, y ejecuta el script

```bash
# Sign in Docker Hub
docker login -u <username>

# Run the script
./docker/build_and_push.sh
```

Creción de un namespace de kuberntes

```bash
kubectl create namespace ml-namespace
```

Creación del archivo `kubernetes/rbac.yaml` que definirá los permisos necesarios

```bash
cat > kubernetes/rbac.yaml << 'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prefect-worker
  namespace: ml-namespace

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: prefect-worker-role
  namespace: ml-namespace
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "pods/status", "services", "configmaps", "secrets"]
    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prefect-worker-rolebinding
  namespace: ml-namespace
subjects:
  - kind: ServiceAccount
    name: prefect-worker
    namespace: ml-namespace
roleRef:
  kind: Role
  name: prefect-worker-role
  apiGroup: rbac.authorization.k8s.io
EOF
```

Creación del worker de prefect

```bash
cat > kubernetes/prefect-worker.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prefect-worker
  namespace: ml-namespace
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prefect-worker
  template:
    metadata:
      labels:
        app: prefect-worker
    spec:
      serviceAccountName: prefect-worker
      containers:
        - name: prefect-worker
          image: yagoutad/ml-training-env:latest
          command:
            - "prefect"
            - "worker"
            - "start"
            - "--pool"
            - "ml-training-pool"
          env:
            - name: PREFECT_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ml-training-secrets
                  key: PREFECT_API_KEY
            - name: DAGSHUB_USERNAME
              valueFrom:
                configMapKeyRef:
                  name: ml-training-config
                  key: DAGSHUB_USERNAME
            - name: DAGSHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: ml-training-secrets
                  key: DAGSHUB_TOKEN
            - name: DAGSHUB_REPO_NAME
              valueFrom:
                configMapKeyRef:
                  name: ml-training-config
                  key: DAGSHUB_REPO_NAME
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
EOF
```

Creación de un ConfigMap y un Secret para las credenciales de DagsHub y Prefect

```bash
cat << EOF > kubernetes/configmap-secrets.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-training-config
  namespace: ml-namespace
data:
  DAGSHUB_USERNAME: "$(grep DAGSHUB_USERNAME .env/.env | cut -d '=' -f2 | tr -d '\"' | tr -d '\n')"
  DAGSHUB_REPO_NAME: "$(grep DAGSHUB_REPO_NAME .env/.env | cut -d '=' -f2 | tr -d '\"' | tr -d '\n')"

---
apiVersion: v1
kind: Secret
metadata:
  name: ml-training-secrets
  namespace: ml-namespace
type: Opaque
data:
  DAGSHUB_TOKEN: "$(grep DAGSHUB_TOKEN .env/.env | cut -d '=' -f2 | tr -d '\"' | tr -d '\n' | base64)"
  PREFECT_API_KEY: "$(grep PREFECT_API_KEY .env/.env | cut -d '=' -f2 | tr -d '\"' | tr -d '\n' | base64)"
EOF
```

Abre una nueba terminal, vete al directorio de trabajo y inicia un prefect server

```bash
prefect server start
```

Vuelve a la primera terminal y crea un pool de trabajo en Prefect

```bash
prefect work-pool create ml-training-pool --type kubernetes
```

Aplica las configuraciones de Kubernetes

```bash
# Primero el RBAC
kubectl apply -f kubernetes/rbac.yaml

# Luego crea el ConfigMap y Secret (después de modificar los valores)
kubectl apply -f kubernetes/configmap-secrets.yaml

# Finalmente, despliega el worker
kubectl apply -f kubernetes/prefect-worker.yaml
```

Verifica que todo está funcionando

```bash
kubectl get pods --namespace ml-namespace
kubectl logs -l app=prefect-worker --namespace ml-namespace
```
