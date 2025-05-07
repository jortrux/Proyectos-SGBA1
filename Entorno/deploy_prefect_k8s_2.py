"""
prefect_k8s_launcher.py

Script automatizador para ejecutar flujos de Prefect en Kubernetes/Minikube.
Uso: python prefect_k8s_launcher.py ruta/al/flujo.py
"""

import argparse
import sys
import os
import logging
import shutil
import subprocess
import time
import uuid
from pathlib import Path
import textwrap
from base64 import b64encode
from typing import Optional, Tuple, Dict, Any, List, Union
import traceback
import platform
import re
from prefect import task, flow, get_run_logger

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prefect_k8s_launcher.log')
    ]
)
global_logger = logging.getLogger("prefect_k8s_launcher")

# Añadir la raíz del proyecto a sys.path
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import *
except ImportError:
    global_logger.error("No se pudo importar la configuración del proyecto. Verificar la estructura del proyecto.")
    sys.exit(1)


IS_WINDOWS = platform.system() == "Windows"


class CommandError(Exception):
    """Excepción personalizada para errores de comandos externos."""
    def __init__(self, cmd: List[str], return_code: int, stdout: str, stderr: str):
        self.cmd = cmd
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        message = f"Comando '{' '.join(cmd)}' falló con código {return_code}.\nStdout: {stdout}\nStderr: {stderr}"
        super().__init__(message)


def parse_arguments() -> argparse.Namespace:
    """
    Analiza los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Objeto con los argumentos parseados.
    
    Raises:
        SystemExit: Si hay errores en los argumentos.
    """
    parser = argparse.ArgumentParser(description='Ejecutar un flujo de Prefect en Kubernetes/Minikube')
    
    # Usar dest para evitar problemas con el guion
    parser.add_argument('flow_script', metavar='flow-script', type=Path, 
                      help='Ruta al script Python que contiene el flujo de Prefect')
    
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument('--requirements', type=Path, default=project_root / 'requirements.txt',
                      help=f'Ruta al archivo requirements.txt (por defecto: PROJECT_ROOT/requirements.txt)')
    parser.add_argument('--data', type=Path, default=project_root / 'data',
                      help=f'Ruta al directorio/archivo de datos (por defecto: PROJECT_ROOT/data)')
    parser.add_argument('--timeout', type=int, default=3600, 
                      help='Tiempo máximo de espera para la ejecución (segundos)')
    parser.add_argument('--image-name', type=str, default='prefect-flow', 
                      help='Nombre base para la imagen Docker')
    parser.add_argument('--base-image', type=str, default='yagoutad/sgba1-base-image:latest', 
                      help='Nombre de la imagen base que usará la imagen del contenedor del pod')
    parser.add_argument('--namespace', type=str, default='default', 
                      help='Namespace de Kubernetes')
    parser.add_argument('--debug', action='store_true', 
                      help='Activar modo de depuración con logs detallados')
    
    args = parser.parse_args()
    
    # Configurar nivel de logging basado en --debug
    if args.debug:
        os.environ["PREFECT_LOGGING_LEVEL"] = "DEBUG"
        global_logger.setLevel(logging.DEBUG)
    
    return args


def run_command(
        cmd: List[str], 
        env: Optional[Dict[str, str]] = None, 
        check: bool = True, 
        capture_output: bool = True,
        shell: bool = False, 
        logger = global_logger
) -> Tuple[int, str, str]:
    """
    Ejecuta un comando externo de forma segura.
    
    Args:
        cmd: Lista con el comando y sus argumentos.
        env: Variables de entorno adicionales.
        check: Si es True, lanza excepción en caso de error.
        capture_output: Si es True, captura y devuelve stdout/stderr.
        shell: Si es True, ejecuta el comando en el shell del sistema.
        logger: loffer utilizado en la función. Por defecto el global_logger.
    
    Returns:
        Tupla con (código_retorno, stdout, stderr).
    
    Raises:
        CommandError: Si check=True y el comando falla.
    """
    logger.debug(f"Ejecutando comando: {' '.join(cmd)}")
    
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)
    
    try:
        # En Windows, algunos comandos pueden requerir shell=True para funcionar
        use_shell = shell or (IS_WINDOWS and cmd[0] in ['minikube', 'docker'])

        process = subprocess.run(
            cmd, 
            env=env_vars, 
            text=True,
            capture_output=capture_output,
            check=False,  # Manejamos el error nosotros
            shell=use_shell
        )
        
        stdout = process.stdout if capture_output else ""
        stderr = process.stderr if capture_output else ""
        
        if process.returncode != 0 and check:
            raise CommandError(cmd, process.returncode, stdout, stderr)
        
        return process.returncode, stdout, stderr
    
    except FileNotFoundError:
        error_msg = f"Comando no encontrado: {cmd[0]}"
        logger.error(error_msg)
        if check:
            raise CommandError(cmd, 127, "", error_msg)
        return 127, "", error_msg


@task
def check_prerequisites() -> None:
    """
    Verifica que estén instalados los prerrequisitos necesarios.
    
    Raises:
        RuntimeError: Si falta algún prerrequisito.
    """
    logger = get_run_logger()

    logger.info("Verificando prerrequisitos...")
    # prereqs = ['docker', 'kubectl', 'prefect', 'minikube']
    prereqs = ['docker', 'kubectl', 'minikube']
    
    missing = []
    for cmd in prereqs:
        # En Windows, buscar también con extensión .exe
        if IS_WINDOWS:
            if shutil.which(cmd) is None and shutil.which(f"{cmd}.exe") is None:
                missing.append(cmd)
        else:
            if shutil.which(cmd) is None:
                missing.append(cmd)
    
    if missing:
        raise RuntimeError(f"No se encontraron los siguientes comandos: {', '.join(missing)}. "
                          f"Asegúrate de tenerlos instalados.")
    
    # Verificar que minikube esté en ejecución
    try:
        shell_param = IS_WINDOWS
        _, stdout, _ = run_command(['minikube', 'status'], shell=shell_param, logger=logger)
        if "Running" not in stdout:
            raise RuntimeError("Minikube no está en ejecución. Inicia minikube con 'minikube start'.")
    except CommandError as e:
        raise RuntimeError(f"Error al verificar minikube: {str(e)}")
    
    logger.info("Todos los prerrequisitos verificados correctamente.")


@task
def setup_dockerfile_dir(
    flow_script_path: Path, 
    requirements_path: Path, 
    base_image_name: str, 
    parent_dockerfile_dir: Path,
    data_path: Optional[Path] = None,
    additional_args: Optional[List[str]] = None
) -> Tuple[str, Path]:
    """
    Crea o actualiza un Dockerfile, copiando en el proceso los archivos necesarios.
    
    Args:
        flow_script_path: Ruta al script de flujo.
        requirements_path: Ruta al archivo de requisitos.
        base_image_name: Nombre de la imagen base.
        parent_dockerfile_dir: Directorio donde crear el directorio con el Dockerfile y sus dependencias.
        data_path: Ruta, opcional, al directorio/archivo de datos.
        additional_args: Lista opcional de argumentos que se pasarán al script de flujo.
    
    Returns:
        Tuple[str, Path]: Tupla con el nombre del flujo (nombre del archivo sin extensión) y la ruta al directorio con el Dockerfile.
    
    Raises:
        FileNotFoundError: Si algún archivo requerido no existe.
    """
    logger = get_run_logger()

    flow_name = flow_script_path.stem

    dockerfile_dir = parent_dockerfile_dir / flow_name

    logger.info(f"Creando entorno en: {dockerfile_dir}")
    
    # Crear directorio si no existe
    dockerfile_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar y copiar el archivo de requirements
    if not requirements_path.exists():
        raise FileNotFoundError(f"El archivo de requirements {requirements_path} no existe")
    
    requirements_dest = dockerfile_dir / 'requirements.txt'
    shutil.copy2(requirements_path, requirements_dest)
    logger.debug(f"Copiado: {requirements_path} -> {requirements_dest}")
    
    copy_data_str = ""
    
    if data_path:
        # Verificar y copiar el directorio/archivo de datos
        if not data_path.exists():
            raise FileNotFoundError(f"El directorio/archivo de datos {data_path} no existe")
        
        data_dest = dockerfile_dir / 'data'
        if data_dest.exists():
            shutil.rmtree(data_dest)
        
        if data_path.is_dir():
            shutil.copytree(data_path, data_dest)
        else:
            data_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data_path, data_dest / data_path.name)
    
        logger.debug(f"Copiado directorio/archivo: {data_path} -> {data_dest}")
        copy_data_str = f"COPY data/ {DOCKER_DATA_DIR}"
    
    # Verificar y copiar el script de flujo
    if not flow_script_path.exists():
        raise FileNotFoundError(f"El script de flujo {flow_script_path} no existe")
    
    flow_dest = dockerfile_dir / 'flow_script.py'
    shutil.copy2(flow_script_path, flow_dest)
    logger.debug(f"Copiado: {flow_script_path} -> {flow_dest}")

    # Si se proporcionan argumentos adicionales, concatenarlos en un string
    args_str = additional_args if additional_args else []
    
    # Crear Dockerfile
    dockerfile_content = textwrap.dedent(
        f"""\
        FROM {base_image_name}

        WORKDIR /opt/prefect

        # Copiar el archivo de dependencias
        COPY requirements.txt .

        # Instalar Prefect y dependencias
        RUN pip install --no-cache-dir -r requirements.txt

        # Copiar los datos
        {copy_data_str}

        # Copiar el script del flujo
        COPY flow_script.py .

        # Ejecutar el flujo cuando se inicie el contenedor
        CMD ["python", "flow_script.py"{f', {', '.join([f'"{arg}"' for arg in args_str])}' if args_str else ""}]
        """
    )
    
    with open(dockerfile_dir / "Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.debug(f"Dockerfile creado en {dockerfile_dir / 'Dockerfile'}")
    return flow_name, dockerfile_dir


def get_minikube_docker_env(logger = global_logger) -> Dict[str, str]:
    """
    Obtiene las variables de entorno para usar el Docker daemon de minikube.

    Args:
        logger: loffer utilizado en la función. Por defecto el global_logger.
    
    Returns:
        Dict[str, str]: Variables de entorno para Docker.
    
    Raises:
        CommandError: Si hay un error al obtener las variables.
    """
    env = os.environ.copy()
    
    if IS_WINDOWS:
        # Para Windows, usar --shell=none para obtener variables en formato clave=valor
        _, minikube_docker_env, _ = run_command(['minikube', 'docker-env', '--shell=none'], logger=logger)
        
        for line in minikube_docker_env.splitlines():
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env[key] = value.strip('"')
    else:
        # Para sistemas Unix
        _, minikube_docker_env, _ = run_command(['minikube', 'docker-env'], logger=logger)
        
        for line in minikube_docker_env.splitlines():
            if line.startswith('export '):
                key, value = line.replace('export ', '').split('=', 1)
                env[key] = value.strip('"')
    
    return env


@task
def build_docker_image(dockerfile_dir: Path, image_name: str) -> str:
    """
    Construye la imagen Docker con el flujo de Prefect.
    
    Args:
        dockerfile_dir: Directorio que contiene el Dockerfile.
        image_name: Nombre base para la imagen.
    
    Returns:
        str: Nombre completo de la imagen construida (con tag).
    
    Raises:
        CommandError: Si hay un error al construir la imagen.
    """
    logger = get_run_logger()

    # Generar un tag único para la imagen
    image_tag = f"{image_name}:{str(uuid.uuid4())[:8]}"
    full_image_name = image_tag
    
    logger.info(f"Construyendo imagen Docker: {full_image_name}")
    
    # Usar el Docker daemon de minikube
    env = get_minikube_docker_env(logger=logger)
    
    # Construir la imagen
    build_cmd = ['docker', 'build', '-t', full_image_name, str(dockerfile_dir)]

    shell_param = IS_WINDOWS
    run_command(build_cmd, env=env, shell=shell_param, logger=logger)
    
    logger.info(f"Imagen Docker construida exitosamente: {full_image_name}")
    return full_image_name


def apply_kubernetes_resource(resource_path: Path, resource_type: str, logger = global_logger) -> None:
    """
    Aplica un recurso de Kubernetes usando kubectl.
    
    Args:
        resource_path: Ruta al archivo YAML del recurso.
        resource_type: Tipo de recurso (ej: "ConfigMap", "Secret", "Job").
        logger: loffer utilizado en la función. Por defecto el global_logger.
    
    Raises:
        CommandError: Si hay un error al aplicar el recurso.
    """
    logger.info(f"Aplicando {resource_type} desde {resource_path}")
    run_command(['kubectl', 'apply', '-f', str(resource_path)], logger=logger)
    logger.info(f"{resource_type} aplicado exitosamente")


@task
def apply_configmap(kubernetes_dir: Path, namespace: str = "default") -> str:
    """
    Crea y aplica el configmap en Kubernetes.
    
    Args:
        kubernetes_dir: Directorio para guardar el archivo YAML.
        namespace: Namespace de Kubernetes.
    
    Returns:
        str: Nombre del configmap creado.
    
    Raises:
        CommandError: Si hay un error al aplicar el configmap.
    """
    logger = get_run_logger()

    logger.info("Creando ConfigMap para Kubernetes...")
    
    # Asegurar que el directorio existe
    kubernetes_dir.mkdir(parents=True, exist_ok=True)
    
    configmap_name = "flow-configmap"
    
    # Variables de entorno para el ConfigMap
    env_vars = {
        "PREFECT_PROFILE": PREFECT_PROFILE,
        "PREFECT_API_URL": PREFECT_API_URL,
        "DAGSHUB_USERNAME": DAGSHUB_USERNAME,
        "DAGSHUB_REPO_NAME": DAGSHUB_REPO_NAME,
        "DOCKER_DATA_DIR": DOCKER_DATA_DIR,
        "KUBERNETES_PV_DIR": KUBERNETES_PV_DIR,
        "REPO_DATA_DIR_PATH": REPO_DATA_DIR_PATH,
        "REPO_MODELS_DIR_PATH": REPO_MODELS_DIR_PATH,
    }
    
    # Crear el YAML para el configmap
    configmap_content = "apiVersion: v1\nkind: ConfigMap\n"
    configmap_content += f"metadata:\n  name: {configmap_name}\n  namespace: {namespace}\n"
    configmap_content += "data:\n"
    
    for key, value in env_vars.items():
        configmap_content += f"  {key}: {value}\n"
    
    # Escribir el YAML
    yaml_path = kubernetes_dir / "kubernetes_configmap.yaml"
    with open(yaml_path, "w") as f:
        f.write(configmap_content)
    
    # Aplicar el configmap
    apply_kubernetes_resource(yaml_path, "ConfigMap", logger=logger)
    
    return configmap_name

@task
def apply_secret(kubernetes_dir: Path, namespace: str = "default") -> str:
    """
    Crea y aplica el Secret en Kubernetes.
    
    Args:
        kubernetes_dir: Directorio para guardar el archivo YAML.
        namespace: Namespace de Kubernetes.
    
    Returns:
        str: Nombre del secret creado.
    
    Raises:
        CommandError: Si hay un error al aplicar el secret.
    """
    logger = get_run_logger()

    logger.info("Creando Secret para Kubernetes...")
    
    # Asegurar que el directorio existe
    kubernetes_dir.mkdir(parents=True, exist_ok=True)
    
    secret_name = "flow-secret"
    
    # Variables para el Secret (codificadas en base64)
    secret_vars = {
        "PREFECT_API_KEY": b64encode(PREFECT_API_KEY.encode()).decode(),
        "DAGSHUB_TOKEN": b64encode(DAGSHUB_TOKEN.encode()).decode(),
    }
    
    # Crear el YAML para el secret
    secret_content = "apiVersion: v1\nkind: Secret\n"
    secret_content += f"metadata:\n  name: {secret_name}\n  namespace: {namespace}\n"
    secret_content += "type: Opaque\ndata:\n"
    
    for key, value in secret_vars.items():
        secret_content += f"  {key}: {value}\n"
    
    # Escribir el YAML
    yaml_path = kubernetes_dir / "kubernetes_secret.yaml"
    with open(yaml_path, "w") as f:
        f.write(secret_content)
    
    # Aplicar el secret
    apply_kubernetes_resource(yaml_path, "Secret", logger=logger)
    
    return secret_name


@task
def apply_pv_pvc(kubernetes_dir: Path, namespace: str = "default") -> Tuple[str, str]:
    """
    Crea y aplica un PV y un PVC en Kubernetes.
    
    Args:
        kubernetes_dir: Directorio para guardar el archivo YAML.
        namespace: Namespace de Kubernetes.
    
    Returns:
        Tuple[str, str]: Nombre del PV y del PVC creados.
    
    Raises:
        CommandError: Si hay un error al aplicar el PV o el PVC.
    """
    logger = get_run_logger()

    logger.info("Creando ConfigMap para Kubernetes...")
    
    # Asegurar que el directorio existe
    kubernetes_dir.mkdir(parents=True, exist_ok=True)
    
    pv_name = "flow-pv"
    pvc_name = "flow-pvc"
    pv_storage = "2Gi"
    pvc_storage = "1Gi"
    
    # Crear el YAML para el configmap
    pv_pvc_yaml = textwrap.dedent(
        f"""\
        # PersistentVolume
        apiVersion: v1
        kind: PersistentVolume
        metadata:
            name: {pv_name}
        spec:
            capacity:
                storage: {pv_storage}
            accessModes:
                - ReadWriteOnce
            storageClassName: manual
            persistentVolumeReclaimPolicy: Delete
            hostPath:
                path: {KUBERNETES_PV_DIR}

        ---

        # PersistentVolumeClaim
        apiVersion: v1
        kind: PersistentVolumeClaim
        metadata:
            name: {pvc_name}
            namespace: {namespace}
        spec:
            accessModes:
                - ReadWriteOnce
            storageClassName: manual
            resources:
                requests:
                    storage: {pvc_storage}
        """
    )
    
    # Escribir el YAML
    yaml_path = kubernetes_dir / "kubernetes_pv_pvc.yaml"
    with open(yaml_path, "w") as f:
        f.write(pv_pvc_yaml)
    
    # Aplicar el configmap
    apply_kubernetes_resource(yaml_path, "PV y PVC", logger=logger)
    
    return pv_name, pvc_name


@task
def deploy_to_kubernetes(
        flow_name: str,
        image_name: str, 
        configmap_name: str,
        secret_name: str, 
        pvc_name: str, 
        kubernetes_dir: Path,
        namespace: str,
        timeout: int
) -> None:
    """
    Despliega el flujo en Kubernetes y crea el job.
    
    Args:
        flow_name: Nombre del flujo.
        image_name: Nombre de la imagen Docker.
        configmap_name: Nombre del ConfigMap.
        secret_name: Nombre del Secret.
        pvc_name: Nombre del PVC.
        kubernetes_dir: Directorio para guardar el archivo YAML.
        namespace: Namespace de Kubernetes.
        timeout: Tiempo máximo de espera (segundos).
    
    Raises:
        RuntimeError: Si el job falla o excede el timeout.
        CommandError: Si hay un error al crear o monitorear el job.
    """
    logger = get_run_logger()

    logger.info(f"Desplegando flujo '{flow_name}' en Kubernetes...")
    
    # Asegurar que el directorio existe
    kubernetes_dir.mkdir(parents=True, exist_ok=True)
    
    # Conseguir un nombre compatible con kubernetes
    flow_job_name = re.sub(r'[^a-zA-Z0-9\-]+', '-', flow_name).lower().strip("-")

    # Crear un nombre único para el job
    job_name = f"{flow_job_name}-{str(uuid.uuid4())[:8]}"
    
    # Crear el YAML para el job
    job_yaml = textwrap.dedent(
        f"""\
        apiVersion: batch/v1
        kind: Job
        metadata:
            name: {job_name}
            namespace: {namespace}
            labels:
                app: prefect-flow
                flow-name: {flow_job_name}
        spec:
            ttlSecondsAfterFinished: 300
            backoffLimit: 0
            template:
                spec:
                    restartPolicy: Never
                    containers:
                      - name: prefect-flow
                        image: {image_name}
                        imagePullPolicy: Never
                        envFrom:
                          - configMapRef:
                                name: {configmap_name}
                          - secretRef:
                                name: {secret_name}
                        volumeMounts:
                          - name: shared-storage
                            mountPath: {KUBERNETES_PV_DIR}
                    volumes:
                      - name: shared-storage
                        persistentVolumeClaim:
                          claimName: {pvc_name}
        """
    )
    
    # Escribir el YAML
    yaml_path = kubernetes_dir / f"kubernetes_{flow_name}_job.yaml"
    with open(yaml_path, "w") as f:
        f.write(job_yaml)
    
    # Aplicar el job
    apply_kubernetes_resource(yaml_path, "Job", logger=logger)
    
    logger.info(f"Monitoreando el estado del job '{job_name}' (timeout: {timeout}s)...")
    monitor_kubernetes_job(job_name, namespace, timeout)


def get_pod_name(job_name: str, namespace: str, logger = global_logger) -> Optional[str]:
    """
    Obtiene el nombre del pod asociado a un job.
    
    Args:
        job_name: Nombre del job.
        namespace: Namespace de Kubernetes.
        logger: loffer utilizado en la función. Por defecto el global_logger.
    
    Returns:
        Optional[str]: Nombre del pod o None si no se encuentra.
    """
    try:
        cmd = [
            'kubectl', 'get', 'pods', 
            '-n', namespace, 
            '-l', f'job-name={job_name}', 
            '-o', 'jsonpath={.items[0].metadata.name}'
        ]
        _, stdout, _ = run_command(cmd, logger=logger)
        pod_name = stdout.strip()
        return pod_name if pod_name else None
    except CommandError:
        return None


def show_pod_logs(pod_name: str, namespace: str, logger = global_logger) -> None:
    """
    Muestra los logs de un pod.
    
    Args:
        pod_name: Nombre del pod.
        namespace: Namespace de Kubernetes.
        logger: loffer utilizado en la función. Por defecto el global_logger.
    """
    if not pod_name:
        logger.warning("No se encontró el pod para mostrar los logs.")
        return
    
    logger.info(f"Mostrando logs del pod {pod_name}...")
    try:
        _, logs, _ = run_command(['kubectl', 'logs', pod_name, '-n', namespace], logger=logger)
        logger.info("=== LOGS DEL POD ===")
        if logs:
            for line in logs.splitlines():
                logger.info(line)
        else:
            logger.warning("No se encontraron logs en el pod.")
        logger.info("====================")
    except CommandError as e:
        logger.error(f"Error al obtener logs: {str(e)}")


@task
def monitor_kubernetes_job(job_name: str, namespace: str, timeout: int) -> None:
    """
    Monitorea el estado de un job de Kubernetes hasta que complete o falle.
    
    Args:
        job_name: Nombre del job.
        namespace: Namespace de Kubernetes.
        timeout: Tiempo máximo de espera (segundos).
    
    Raises:
        RuntimeError: Si el job falla o excede el timeout.
    """
    logger = get_run_logger()

    start_time = time.time()
    check_interval = 10  # segundos entre verificaciones
    
    while time.time() - start_time < timeout:
        # Verificar estado del job (completado)
        try:
            cmd = [
                'kubectl', 'get', 'job', job_name, 
                '-n', namespace, 
                '-o', 'jsonpath={.status.conditions[?(@.type=="Complete")].status}'
            ]
            _, complete_status, _ = run_command(cmd, logger=logger)
            
            if complete_status.strip() == "True":
                logger.info(f"Job completado exitosamente")
                pod_name = get_pod_name(job_name, namespace, logger=logger)
                show_pod_logs(pod_name, namespace, logger=logger)
                return
            
            # Verificar si falló
            cmd = [
                'kubectl', 'get', 'job', job_name, 
                '-n', namespace, 
                '-o', 'jsonpath={.status.conditions[?(@.type=="Failed")].status}'
            ]
            _, failed_status, _ = run_command(cmd, logger=logger)
            
            if failed_status.strip() == "True":
                logger.error(f"Job falló")
                pod_name = get_pod_name(job_name, namespace, logger=logger)
                show_pod_logs(pod_name, namespace, logger=logger)
                raise RuntimeError(f"El job '{job_name}' falló. Revisa los logs para más detalles.")
            
            # También verificar si hay pods fallidos
            pod_name = get_pod_name(job_name, namespace, logger=logger)
            if pod_name:
                cmd = [
                    'kubectl', 'get', 'pod', pod_name, 
                    '-n', namespace, 
                    '-o', 'jsonpath={.status.phase}'
                ]
                _, pod_status, _ = run_command(cmd, logger=logger)
                
                if pod_status.strip() == "Failed":
                    logger.error(f"Pod falló: {pod_name}")
                    show_pod_logs(pod_name, namespace, logger=logger)
                    raise RuntimeError(f"El pod '{pod_name}' falló. Revisa los logs para más detalles.")
        
        except CommandError as e:
            logger.warning(f"Error al verificar estado del job: {str(e)}")
        
        # Esperar antes de verificar de nuevo
        logger.info(f"Job en progreso, esperando {check_interval}s...")
        time.sleep(check_interval)
    
    # Si llegamos aquí, se excedió el timeout
    logger.error(f"Timeout excedido ({timeout}s)")
    pod_name = get_pod_name(job_name, namespace, logger=logger)
    show_pod_logs(pod_name, namespace, logger=logger)
    raise RuntimeError(f"El job '{job_name}' excedió el tiempo de espera de {timeout}s.")


@flow
def main() -> int:
    """
    Función principal del script.
    
    Returns:
        int: Código de salida (0 si todo fue exitoso, 1 en caso de error).
    """
    logger = get_run_logger()

    try:
        args = parse_arguments()
        script_dir = Path(__file__).resolve().parent
        
        # Verificar prerrequisitos
        check_prerequisites()
        
        # Preparar directorio para Docker
        parent_dockerfile_dir = script_dir / 'docker' / 'app'
        flow_name, dockerfile_dir = setup_dockerfile_dir(
            flow_script_path=args.flow_script.resolve(), 
            requirements_path=args.requirements.resolve(),
            base_image_name=args.base_image,
            parent_dockerfile_dir=parent_dockerfile_dir,
            data_path=args.data.resolve()
        )
        
        # Construir imagen Docker
        image_name = build_docker_image(dockerfile_dir, args.image_name)
        
        # Preparar directorio para Kubernetes
        kubernetes_dir = script_dir / 'kubernetes'
        
        # Aplicar recursos de Kubernetes
        configmap_name = apply_configmap(
            kubernetes_dir=kubernetes_dir,
            namespace=args.namespace
        )
        
        secret_name = apply_secret(
            kubernetes_dir=kubernetes_dir,
            namespace=args.namespace
        )

        pv_name, pvc_name = apply_pv_pvc(
            kubernetes_dir=kubernetes_dir,
            namespace=args.namespace
        )
        
        # Desplegar en Kubernetes
        deploy_to_kubernetes(
            flow_name=flow_name,
            image_name=image_name,
            configmap_name=configmap_name,
            secret_name=secret_name,
            pvc_name=pvc_name,
            kubernetes_dir=kubernetes_dir,
            namespace=args.namespace,
            timeout=args.timeout
        )
        
        logger.info("\nPuedes ver los resultados detallados en Prefect Cloud")
        logger.info(f"URL de Prefect: {PREFECT_API_URL}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())

