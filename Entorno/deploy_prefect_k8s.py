from prefect import flow, task
from prefect.logging import get_run_logger
import tempfile
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import shutil
from base64 import b64encode
import time
from pathlib import Path
import sys
import os

# Añadir la raíz del proyecto a sys.path
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import *
except ImportError:
    print("No se pudo importar la configuración del proyecto. Verificar la estructura del proyecto.")
    sys.exit(1)

def run_command(command: str) -> str:
    """Ejecuta un comando en la terminal y maneja errores"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error ejecutando '{command}': {e.stderr}")
    
def check_pod_exists(core_v1, namespace: str, job_name: str) -> Optional[str]:
    """
    Verifica si existe el pod y retorna su nombre
    """

    pods = core_v1.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"job-name={job_name}"
    )
    
    if pods.items:
        return pods.items[0].metadata.name
    return None

@task
def validate_environment() -> bool:
    """Verifica que las herramientas necesarias estén instaladas"""
    logger = get_run_logger()
    required_commands = ['docker', 'kubectl', 'minikube']
    for cmd in required_commands:
        if not shutil.which(cmd):
            raise RuntimeError(f"No se encontró {cmd}. Por favor, instálalo.")
    
    run_command("minikube status")
    logger.info("✅ Ambiente validado correctamente")
    return True

@task
def create_Dockerfile(script_path: Path, requirements_path: Path = PROJECT_ROOT / "requirements.txt", data_dir_path: Path = PROJECT_ROOT / "data") -> Tuple[Path, Path]:
    """Prepara el Dockerfile para crear la imagen de Docker"""
    logger = get_run_logger()
    temp_dir = Path(tempfile.mkdtemp())
    dockerfile_path = temp_dir / "Dockerfile"
    
    dockerfile_content = [
        "FROM prefecthq/prefect:3-python3.12",
        "WORKDIR /opt/prefect",
    ]
    
    if requirements_path and requirements_path.exists():
        shutil.copy2(requirements_path, temp_dir / "requirements.txt")
        dockerfile_content.extend(["COPY requirements.txt .", "RUN pip install -r requirements.txt"])
    
    if data_dir_path:
        shutil.copytree(data_dir_path, temp_dir / "data", dirs_exist_ok=True)
        dockerfile_content.extend(["COPY data ./data"])

    dockerfile_content.extend(["COPY flow.py ."])
    
    dockerfile_content.extend(['CMD ["python", "flow.py"]'])

    dockerfile_path.write_text("\n".join(dockerfile_content))
    shutil.copy2(script_path, temp_dir / "flow.py")

    logger.info("✅ Dockerfile preparado")
    return temp_dir, dockerfile_path

@task
def create_kubernetes_secret(namespace: str = "default") -> str:
    """Crea o actualiza un Secret en Kubernetes"""
    logger = get_run_logger()
    secret_name = "flow-secret"

    try:
        # Cargar configuración del cluster
        config.load_kube_config()
        
        # Crear cliente de la API
        v1 = client.CoreV1Api()

        # Crear objeto Secret
        secret = client.V1Secret(
            api_version="v1",
            kind="Secret",
            metadata=client.V1ObjectMeta(
                name=secret_name,
                namespace=namespace
            ),
            type="Opaque",
            data={
                "PREFECT_API_KEY": b64encode(PREFECT_API_KEY.encode()).decode(),
                "DAGSHUB_TOKEN": b64encode(DAGSHUB_TOKEN.encode()).decode()
            }
        )

        try:
            # Intentar crear el Secret
            v1.create_namespaced_secret(namespace=namespace, body=secret)
            logger.info("✅ Secret creado en Kubernetes")
        except ApiException as e:
            if e.status == 409:  # Código 409 indica que el recurso ya existe
                v1.replace_namespaced_secret(name=secret_name, namespace=namespace, body=secret)
                logger.info("🔄 Secret existente actualizado en Kubernetes")
            else:
                raise
    
    except Exception as e:
        logger.error(f"❌ Error al crear/actualizar el Secret: {str(e)}")
        raise RuntimeError(f"Error al crear/actualizar el Secret: {str(e)}")
    
    return secret_name
    
@task
def create_kubernetes_configmap(namespace: str = "default") -> str:
    """Crea o actualiza un ConfigMap en Kubernetes"""
    logger = get_run_logger()
    configmap_name = "flow-configmap"

    try:
        # Cargar configuración del cluster
        config.load_kube_config()
        
        # Crear cliente de la API
        v1 = client.CoreV1Api()

        # Crear objeto ConfigMap
        configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(
                name=configmap_name,
                namespace=namespace,
            ),
            data={
                "PREFECT_PROFILE": PREFECT_PROFILE,
                "PREFECT_API_URL": PREFECT_API_URL,
                "DAGSHUB_USERNAME": DAGSHUB_USERNAME,
                "DAGSHUB_REPO_NAME": DAGSHUB_REPO_NAME,
                "REPO_DATA_DIR_PATH": REPO_DATA_DIR_PATH,
                "REPO_MODELS_DIR_PATH": REPO_MODELS_DIR_PATH
            }
        )

        try:
            # Intentar crear el ConfigMap
            v1.create_namespaced_config_map(namespace=namespace, body=configmap)
            logger.info("✅ ConfigMap creado en Kubernetes")
        except ApiException as e:
            if e.status == 409:  # Código 409 indica que el recurso ya existe
                v1.replace_namespaced_config_map(name=configmap_name, namespace=namespace, body=configmap)
                logger.info("🔄 ConfigMap existente actualizado en Kubernetes")
            else:
                raise
    
    except Exception as e:
        logger.error(f"❌ Error al crear/actualizar el ConfigMap: {str(e)}")
        raise RuntimeError(f"Error al crear/actualizar el ConfigMap: {str(e)}")
    
    return configmap_name

@task
def build_docker_image(dockerfile_dir: Path, dockerfile_Path: Path) -> str:
    """Construye la imagen Docker"""
    logger = get_run_logger()
    image_name = "yagoutad/prefect-flow:latest"
    run_command(f"docker build -f {dockerfile_Path} -t {image_name} {dockerfile_dir}")
    run_command(f"docker push {image_name}")
    logger.info("✅ Imagen Docker construida y subida")
    return image_name

@task
def create_kubernetes_job(image_name: str, configmap_name: str, secret_name: str, namespace: str = "default") -> str:
    """Crea o actualiza un Job en Kubernetes"""
    logger = get_run_logger()
    job_name = "flow-job"

    try:
        # Cargar configuración del cluster
        config.load_kube_config()
        
        # Crear cliente de la API
        v1 = client.BatchV1Api()

        # Eliminar el Job si ya existe
        try:
            v1.delete_namespaced_job(name=job_name, namespace=namespace, propagation_policy="Foreground")
        except ApiException as e:
            if e.status != 404:  # Si el Job no existe, ignoramos el error 404
                raise

        # Preparar contenedor
        container = client.V1Container(
            name="job-container",
            image=image_name,
            # image_pull_policy="IfNotPresent",
            image_pull_policy="Always",
            env_from=[
                # Cargar todas las variables desde el ConfigMap
                client.V1EnvFromSource(
                    config_map_ref=client.V1ConfigMapEnvSource(name=configmap_name)
                ),
                # Cargar todas las variables desde el Secret
                client.V1EnvFromSource(
                    secret_ref=client.V1SecretEnvSource(name=secret_name)
                )
            ]
        )
        
        # Crear template del Pod
        template = client.V1PodTemplateSpec(
            spec=client.V1PodSpec(
                containers=[container],
                restart_policy="Never"
            )
        )
        
        # Crear especificación del Job
        job_spec = client.V1JobSpec(
            template=template,
            backoff_limit=0
        )
        
        # Crear objeto Job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=namespace
            ),
            spec=job_spec
        )

        time.sleep(2)
        try:
            # Intentar crear el Job
            v1.create_namespaced_job(namespace=namespace, body=job)
            logger.info("✅ Job creado en Kubernetes")
        except ApiException as e:
            if e.status == 409:  # Código 409 indica que el todavía existe
                time.sleep(5)
                try:
                    # Intentar crear el Job
                    v1.create_namespaced_job(namespace=namespace, body=job)
                    logger.info("✅ Job creado en Kubernetes")
                except ApiException as e:
                    raise
            else:
                raise
    
    except Exception as e:
        logger.error(f"❌ Error al crear/actualizar el Job: {str(e)}")
        raise RuntimeError(f"Error al crear/actualizar el Job: {str(e)}")
    
    return job_name

@task(retries=40, retry_delay_seconds=3)  # 2 minutos de timeout, chequea cada 3 segundos
def get_job_pod(job_name: str, namespace: str = "default") -> str:
    """Task que espera y obtiene el nombre del pod creado por un Job"""
    logger = get_run_logger()
    
    try:
        config.load_kube_config()
        v1 = client.CoreV1Api()
        
        logger.info(f"🔍 Buscando pod para el job {job_name}")
        pod_name = check_pod_exists(v1, namespace, job_name)
        
        if pod_name:
            logger.info(f"✅ Pod encontrado: {pod_name}")
            return pod_name
            
        raise ValueError("Pod no encontrado")
        
    except Exception as e:
        logger.error(f"❌ Error al buscar el pod: {str(e)}")
        raise

@task
def cleanup(temp_dir: Path):
    """Limpia archivos temporales"""
    logger = get_run_logger()
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    logger.info("✅ Limpieza completada")


@flow(name="deploy-prefect-to-k8s")
def deploy_flow_to_kubernetes(script_path: str, namespace: str = "default", requirements_path: str =str(PROJECT_ROOT / 'requirements.txt'), data_dir_path: str = str(PROJECT_ROOT / 'data')):
    logger = get_run_logger()
    script_path = Path(script_path)
    requirements_path = Path(requirements_path) if requirements_path else None
    validate_environment()
    secret_name = create_kubernetes_secret(namespace)
    configmap_name = create_kubernetes_configmap(namespace)
    temp_dir, dockerfile_path = create_Dockerfile(script_path, requirements_path, data_dir_path)
    image_name = build_docker_image(temp_dir, dockerfile_path)
    job_name = create_kubernetes_job(image_name, configmap_name, secret_name, namespace)
    pod_name = get_job_pod(job_name, namespace)
    cleanup(temp_dir)
    logger.info(f"\n🚀 Despliegue completado!\n Para ver el job ejecute:\n kubectl logs -f {pod_name} --namespace {namespace}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Despliega un flujo de Prefect en Kubernetes')
    parser.add_argument('script_path', type=str, help='Ruta al script de Python con el flujo de Prefect')
    parser.add_argument('--namespace', type=str, default='default', help='Namespace de Kubernetes (opcional)')
    parser.add_argument('--requirements', type=str, default=str(PROJECT_ROOT / 'requirements.txt'), help='Ruta al archivo requirements.txt (opcional)')
    parser.add_argument('--data', type=str, default=str(PROJECT_ROOT / 'data'), help='Ruta al directorio de datos (opcional)')
    args = parser.parse_args()
    deploy_flow_to_kubernetes(script_path=args.script_path, namespace=args.namespace, requirements_path=args.requirements, data_dir_path=args.data)
