#!/bin/bash

set -e  # Para salir si hay errores

echo "🧹 [1/6] Eliminando imagen local antigua (opcional)..."
docker rmi joanbg/mi-fastapi-app:latest || echo "⚠️ Imagen no estaba presente localmente"

echo "🛠️  [2/6] Reconstruyendo imagen Docker..."
docker build -t joanbg/mi-fastapi-app:latest .

echo "📤 [3/6] Subiendo nueva imagen a Docker Hub..."
docker push joanbg/mi-fastapi-app:latest

echo "🗑️  [4/6] Eliminando deployment anterior (opcional)..."
kubectl delete deployment fastapi-deployment || echo "⚠️ No existía deployment previo"

echo "🚀 [5/6] Aplicando nuevo deployment..."
kubectl apply -f k8s/deployment.yaml

echo "⏳ Esperando a que los pods estén listos..."
kubectl wait --for=condition=ready pod -l app=fastapi --timeout=60s

echo "🌍 URL disponible para Postman o navegador:"
minikube service fastapi-service --url
