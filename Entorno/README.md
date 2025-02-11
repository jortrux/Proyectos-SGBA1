# Entorno MLOps Compartido

## Requisitos previos
- Linux (preferiblemente Ubuntu/Mint)
- Acceso a Internet
- Al menos 8GB de RAM
- 20GB de espacio libre en disco

## Pasos de instalación

1. Clonar el repositorio:
```bash
git clone <URL_DEL_REPO>
cd mlops-project
```

2. Crear archivo .env con las credenciales compartidas:
```bash
DAGSHUB_USERNAME=usuario_compartido
DAGSHUB_PASSWORD=contraseña_compartida
GOOGLE_DRIVE_CREDENTIALS=credenciales_compartidas
```

3. Ejecutar el script de configuración:
```bash
chmod +x setup.sh
./setup.sh
```

## Uso del entorno

1. Verificar que todo está funcionando:
```bash
kubectl get pods -n mlops
```

2. Acceder al entorno:
```bash
kubectl exec -it -n mlops deployment/mlops-training -- bash
```

3. Acceder a las interfaces web:
- Prefect UI: http://localhost:8000
- DagsHub: http://localhost:4200

## Solución de problemas comunes

- Si los pods no arrancan, verificar:
  ```bash
  kubectl describe pod -n mlops
  ```
- Si hay problemas con Docker:
  ```bash
  docker ps
  docker logs <container-id>
  ```
  