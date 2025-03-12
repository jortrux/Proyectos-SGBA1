# Entorno de Despliegue para Flujos Prefect en Kubernetes

Este directorio contiene las herramientas necesarias para construir y desplegar flujos de Prefect en un entorno Kubernetes, específicamente usando Minikube para desarrollo local.

## Estructura del Directorio

```
Entorno
├── deploy_prefect_k8s.py          # Script principal para desplegar flujos
└── docker
    └── base
        ├── build_push.py          # Script para crear y publicar la imagen base
        └── Dockerfile             # Definición de la imagen base
```

## Requisitos Previos

### Software Necesario

Antes de utilizar este entorno, asegúrate de tener instalado:

1. **Python 3.9+**
   - [Descargar Python](https://www.python.org/downloads/)

2. **Docker**
   - [Instalación para Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Instalación para Linux](https://docs.docker.com/engine/install/)
   - En Linux, después de la instalación, añade tu usuario al grupo docker para ejecutar comandos sin sudo:

     ```bash
     sudo usermod -aG docker $USER
     # Reinicia tu sesión después
     ```

3. **Kubernetes CLI (kubectl)**
   - [Instalación de kubectl](https://kubernetes.io/docs/tasks/tools/)

4. **Minikube**
   - [Instalación de Minikube](https://minikube.sigs.k8s.io/docs/start/)

5. **Prefect**
   - Instalación: `pip install -U prefect`
   - [Documentación de Prefect](https://docs.prefect.io/)

### Configuración del Entorno

1. Clona el repositorio:

   ```bash
   git clone https://github.com/jortrux/Proyectos-SGBA1.git
   cd Proyectos-SGBA1
   ```

2. Instala las dependencias del proyecto:

   ```bash
   pip install -r requirements.txt
   ```

3. Asegúrate de tener configuradas las variables de entorno necesarias en el archivo `config.py` en un archivo `.env` la raíz del proyecto:
   - `PREFECT_PROFILE`
   - `PREFECT_API_URL`
   - `PREFECT_API_KEY`
   - `DAGSHUB_USERNAME`
   - `DAGSHUB_REPO_NAME`
   - `DAGSHUB_TOKEN`
   - `DOCKER_DATA_DIR`
   - `REPO_DATA_DIR_PATH`
   - `REPO_MODELS_DIR_PATH`

## Instrucciones de Uso

### 1. Preparar la Imagen Base (Opcional)

La imagen base contiene todas las dependencias comunes para tus flujos de Prefect. Esta imagen debe ser creada y publicada antes de desplegar flujos específicos.

```bash
# Navega al directorio de la imagen base
cd Entorno/docker/base

# Construye y publica la imagen base
python build_push.py

# Opcionalmente, si solo quieres construir sin publicar
python build_push.py --only-build
```

Opciones adicionales:

- `--requirements RUTA`: Especifica una ruta alternativa al archivo requirements.txt
- `--repository NOMBRE`: Cambia el nombre del repositorio Docker (por defecto: yagoutad/sgba1-base-image)
- `--verbose` o `-v`: Muestra información detallada durante la ejecución

### 2. Iniciar Minikube

Antes de desplegar cualquier flujo, asegúrate de que Minikube esté en funcionamiento:

```bash
# Inicia Minikube (preferiblemente con el driver de Docker)
minikube start --driver=docker

# Verifica que está funcionando correctamente
minikube status
```

### 3. Desplegar un Flujo de Prefect

Una vez que la imagen base esté preparada y Minikube esté en ejecución, puedes desplegar tus flujos de Prefect:

```bash
# Desde la raíz del repositorio
python Entorno/deploy_prefect_k8s.py ruta/a/tu/flujo.py
```

Opciones adicionales:

- `--requirements RUTA`: Especifica una ruta alternativa al archivo requirements.txt
- `--data RUTA`: Ruta al directorio de datos que necesita el flujo (por defecto: `PROJECT_ROOT/data`)
- `--timeout SEGUNDOS`: Tiempo máximo de espera para la ejecución (por defecto: 3600s)
- `--image-name NOMBRE`: Nombre base para la imagen Docker
- `--base-image IMAGEN`: Nombre de la imagen base a utilizar (por defecto: yagoutad/sgba1-base-image:latest)
- `--namespace NAMESPACE`: Namespace de Kubernetes (por defecto: default)
- `--debug`: Activa el modo de depuración con logs detallados

## Monitoreo y Logs

Durante la ejecución de un flujo, puedes monitorear su progreso:

1. A través de la interfaz de Prefect Cloud:
   - Accede a la URL especificada en `PREFECT_API_URL`

2. Mediante los logs generados:
   - Los logs se muestran en la consola durante la ejecución
   - También se guardan en el archivo `prefect_k8s_launcher.log`

## Notas Adicionales

- El script `deploy_prefect_k8s.py` gestiona automáticamente la creación de ConfigMaps y Secrets en Kubernetes para manejar variables de entorno sensibles.
- Todos los artefactos temporales (Dockerfiles, manifiestos de Kubernetes) se generan en subdirectorios dentro de `Entorno/`.
- Para depurar problemas, utiliza la opción `--debug` para obtener logs más detallados.

## Limpieza

Después de usar el entorno, puedes limpiar los recursos:

```bash
# Para detener Minikube
minikube stop

# Para eliminar completamente el clúster de Minikube
minikube delete
```
