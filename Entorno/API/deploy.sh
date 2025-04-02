#!/bin/bash

echo "▶️  Construyendo imagen Docker..."
docker build -t joanbg/mi-fastapi-app:latest .

echo "📤 Subiendo imagen a Docker Hub..."
docker push joanbg/mi-fastapi-app:latest

echo "♻️  Reiniciando despliegue en Kubernetes..."
kubectl rollout restart deployment fastapi-deployment

echo "✅ Esperando pods listos..."
kubectl wait --for=condition=ready pod -l app=fastapi --timeout=60s

echo "🌐 Obteniendo URL de acceso a FastAPI..."
minikube service fastapi-service --url
