from prefect import flow, task
from prefect.logging import get_run_logger
import tempfile
import yaml
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict
import shutil
from dotenv import load_dotenv
import os

@task
def create_kubernetes_secret(env_file: Path) -> str:
    """Crea o actualiza el secret en Kubernetes utilizando el archivo .env"""
    logger = get_run_logger()

    secret_name = "flow-secret"
    
    if not env_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo .env en: {env_file}")
    
    delete_command = f"kubectl delete secret {secret_name} --ignore-not-found=true"
    create_command = f"kubectl create secret generic {secret_name} --from-env-file={env_file.as_posix()}"

    try:
        # Ejecutar el comando para eliminar el Secret (si existe)
        subprocess.run(delete_command, shell=True, check=True)
        # Intentar crear el secret
        result = subprocess.run(
            create_command,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("✅ Secret creado/actualizado exitosamente")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al intentar crear el 'Secret': {e.stderr}")
    
    return secret_name

@task
def validate_environment() -> bool:
    """Verifica que el ambiente tenga todas las herramientas necesarias"""
    logger = get_run_logger()
    required_commands = ['docker', 'kubectl', 'minikube']
    
    for cmd in required_commands:
        if not shutil.which(cmd):
            raise RuntimeError(f"No se encontró {cmd}. Por favor, instálalo.")
    
    # Verificar que minikube está corriendo
    try:
        subprocess.run(['minikube', 'status'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("Minikube no está corriendo. Ejecuta 'minikube start'")
    
    logger.info("✅ Ambiente validado correctamente")
    return True

@task
def prepare_deployment_files(
    script_path: Path,
    requirements_path: Optional[Path],
    secret_name: str
) -> Tuple[Path, Path, Path]:
    """Prepara los archivos necesarios para el despliegue"""
    logger = get_run_logger()
    
    # Crear directorio temporal
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Creando archivos en directorio temporal: {temp_dir}")
    
    # Crear Dockerfile
    dockerfile_content = [
        "FROM prefecthq/prefect:3-python3.12",
        "WORKDIR /opt/prefect",
    ]
    
    if requirements_path and requirements_path.exists():
        shutil.copy2(requirements_path, temp_dir / "requirements.txt")
        dockerfile_content.extend([
            "COPY requirements.txt .",
            "RUN pip install -r requirements.txt"
        ])
    
    dockerfile_content.extend([
        "COPY flow.py .",
        'CMD ["python", "flow.py"]'
    ])
    
    dockerfile_path = temp_dir / "Dockerfile"
    dockerfile_path.write_text("\n".join(dockerfile_content))
    
    # Copiar script de flujo
    flow_path = temp_dir / "flow.py"
    shutil.copy2(script_path, flow_path)
    
    # Crear YAML del Job
    job_yaml = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "prefect-flow-job"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "prefect-flow-container",
                        "image": "yagoutad/prefect-flow:latest",
                        "imagePullPolicy": "Never",
                        "envFrom": [{
                            "secretRef": {"name": secret_name}
                        }]
                    }],
                    "restartPolicy": "Never"
                }
            },
            "backoffLimit": 0
        }
    }
    
    job_yaml_path = temp_dir / "job.yaml"
    job_yaml_path.write_text(yaml.dump(job_yaml))
    
    logger.info("✅ Archivos de despliegue preparados")
    return temp_dir, dockerfile_path, job_yaml_path

@task
def build_docker_image(dockerfile_dir: Path) -> str:
    """Construye la imagen Docker"""
    logger = get_run_logger()
    image_name = "yagoutad/prefect-flow:latest"
    
    logger.info("Construyendo imagen Docker...")
    result = subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        cwd=dockerfile_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Error al construir imagen Docker: {result.stderr}")
    
    subprocess.run(
        ["docker", "push", image_name],
        cwd=dockerfile_dir,
        capture_output=True,
        text=True
    )
    
    logger.info("✅ Imagen Docker construida exitosamente")
    return image_name

@task
def deploy_kubernetes_job(job_yaml_path: Path) -> str:
    """Despliega el Job en Kubernetes"""
    logger = get_run_logger()
    
    logger.info("Desplegando Job en Kubernetes...")
    result = subprocess.run(
        ["kubectl", "apply", "-f", str(job_yaml_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Error al desplegar Job: {result.stderr}")
    
    # Obtener el nombre del pod
    pod_name = subprocess.run(
        ["kubectl", "get", "pods", "--selector=job-name=prefect-flow-job", "-o", "jsonpath='{.items[0].metadata.name}'"],
        capture_output=True,
        text=True
    ).stdout.strip("'")
    
    logger.info("✅ Job desplegado exitosamente")
    return pod_name

@task
def cleanup(temp_dir: Path):
    """Limpia los archivos temporales"""
    logger = get_run_logger()
    shutil.rmtree(temp_dir)
    logger.info("✅ Limpieza completada")

@flow(name="deploy-prefect-to-k8s")
def deploy_flow_to_kubernetes(
    script_path: str,
    env_file: str,
    requirements_path: Optional[str] = None,
):
    """
    Flujo principal que despliega un script de Prefect en Kubernetes
    
    Args:
        script_path: Ruta al script de Python con el flujo de Prefect
        env_file: Ruta al archivo .env con las variables de entorno
        requirements_path: Ruta opcional al archivo requirements.txt
    """
    logger = get_run_logger()
    
    # Convertir paths a objetos Path
    script_path = Path(script_path)
    env_file = Path(env_file)
    requirements_path = Path(requirements_path) if requirements_path else None
    
    # Validar que los archivos existen
    if not script_path.exists():
        raise FileNotFoundError(f"No se encontró el script: {script_path}")
    
    # Ejecutar el flujo
    validate_environment()
    secret_name = create_kubernetes_secret(env_file)
    temp_dir, dockerfile_path, job_yaml_path = prepare_deployment_files(
        script_path,
        requirements_path,
        secret_name
    )
    image_name = build_docker_image(temp_dir)
    pod_name = deploy_kubernetes_job(job_yaml_path)
    cleanup(temp_dir)
    
    logger.info(f"""
    🚀 Despliegue completado exitosamente!
    
    Para monitorear tu Job:
        kubectl get jobs
        kubectl get pods
        kubectl logs -f {pod_name}
    """)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Despliega un flujo de Prefect en Kubernetes')
    parser.add_argument('script_path', type=str, help='Ruta al script de Python con el flujo de Prefect')
    parser.add_argument('--env-file', type=str, default='.env', help='Ruta al archivo .env (por defecto: .env)')
    parser.add_argument('--requirements', type=str, help='Ruta al archivo requirements.txt (opcional)')
    
    args = parser.parse_args()
    
    deploy_flow_to_kubernetes(
        script_path=args.script_path,
        env_file=args.env_file,
        requirements_path=args.requirements
    )