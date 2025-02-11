#!/bin/bash

# Colores para los mensajes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Iniciando configuración del entorno MLOps...${NC}"

# 1. Verificar/Instalar dependencias
echo -e "${GREEN}1. Instalando dependencias...${NC}"
sudo apt-get update
sudo apt-get install -y docker.io curl git python3-pip

# 2. Instalar Kind
echo -e "${GREEN}2. Instalando Kind...${NC}"
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.26.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/

# 3. Instalar kubectl
echo -e "${GREEN}3. Instalando kubectl...${NC}"
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# 4. Crear directorios necesarios
echo -e "${GREEN}4. Creando estructura de directorios...${NC}"
mkdir -p shared-data
mkdir -p kubernetes-config

# 5. Crear cluster de Kind
echo -e "${GREEN}5. Creando cluster de Kubernetes...${NC}"
kind create cluster --config kind-config.yaml

# 6. Construir y cargar la imagen Docker
echo -e "${GREEN}6. Construyendo imagen Docker...${NC}"
docker build -t mlops-training:latest .
kind load docker-image mlops-training:latest

# 7. Aplicar configuraciones de Kubernetes
echo -e "${GREEN}7. Aplicando configuraciones de Kubernetes...${NC}"
kubectl apply -f kubernetes-config/

# 8. Instalar DVC
echo -e "${GREEN}8. Instalando DVC...${NC}"
pip install 'dvc[gdrive]'

echo -e "${BLUE}Configuración completada!${NC}"
echo -e "${GREEN}Para acceder al entorno:${NC}"
echo "1. Prefect UI: http://localhost:8000"
echo "2. DagsHub: http://localhost:4200"
echo -e "${GREEN}Para verificar el estado:${NC}"
echo "kubectl get pods"