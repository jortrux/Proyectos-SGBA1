
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
from prefect import task, flow, get_run_logger
import pandas as pd
from datetime import timedelta

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_pipeline_launcher.log')
    ]
)
global_logger = logging.getLogger("training_pipeline_launcher")

# Añadir la raíz del proyecto a sys.path
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

    # Importar funciones y variables de entrono de otros scripts
    from config import *
    from Entorno.deploy_prefect_k8s_2 import check_prerequisites, setup_dockerfile_dir, build_docker_image, apply_configmap, apply_secret, apply_pv_pvc, deploy_to_kubernetes, run_command
except ImportError:
    global_logger.error("Errores al intentar hacer las distintas importaciones. Verificar la estructura del proyecto.")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Analiza los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Objeto con los argumentos parseados.
    
    Raises:
        SystemExit: Si hay errores en los argumentos.
    """
    parser = argparse.ArgumentParser(description='Ejecutar un entrenamiento de los 3 modelos en Kubernetes/Minikube')
    
    project_root = Path(__file__).resolve().parent.parent.parent

    # Usar dest para evitar problemas con el guion
    parser.add_argument('--precio', type=Path, default=project_root / "src" / "prefect_flows" / "precio_flow.py",
                      help=f'Ruta al script Python que contiene el flujo de entrenamiento del precio')
    parser.add_argument('--consumo', type=Path, default=project_root / "src" / "prefect_flows" / "consumo_flow.py",
                      help=f'Ruta al script Python que contiene el flujo de entrenamiento del consumo')
    parser.add_argument('--agente', type=Path, default=project_root / "src" / "prefect_flows" / "agente_flow.py",
                      help=f'Ruta al script Python que contiene el flujo de entrenamiento del agente')
    
    parser.add_argument('--requirements', type=Path, default=project_root / 'requirements.txt',
                      help=f'Ruta al archivo requirements.txt (por defecto: PROJECT_ROOT/requirements.txt)')
    parser.add_argument('--data-precio', type=Path, default=project_root / 'data/processed/datos_precio/clima_precio_merged_recortado.parquet',
                      help=f'Ruta a los datos de entrenamiento de precio (por defecto: PROJECT_ROOT/data/processed/datos_precio/clima_precio_merged_recortado.parquet)')
    parser.add_argument('--data-consumo', type=Path, default=project_root / 'data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv',
                      help=f'Ruta a los datos de entrenamiento de consumo (por defecto: PROJECT_ROOT/data/processed/datos_consumo/hogar_individual_bcn/casa_bcn_clean.csv)')
    parser.add_argument('--timeout', type=int, default=3600, 
                      help='Tiempo máximo de espera para la ejecución (segundos)')
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


@task
def get_results(pv_name: str, pvc_name: str, namespace: str = "default") -> None:

    logger = get_run_logger()

    # Crear carpeta donde guardar resultados
    results_path = Path(__file__).resolve().parent / "simulation_results"
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Carpeta de resultados creada en {results_path}")

    # Guardar datos de la simulación en directorio temporal
    run_command(["minikube", "ssh", "--", "sudo", "mkdir", "-p", "/tmp/data"], logger=logger)
    run_command(["minikube", "ssh", "--", "sudo", "cp", "-r", f"{KUBERNETES_PV_DIR}*", "/tmp/data"], logger=logger)
    
    # Guardar los datos de minikube en local
    run_command(["minikube", "ssh", "--", "sudo", "tar", "-czf", "/tmp/data.tar.gz", "-C", "/tmp", "data"], logger=logger)
    run_command(["minikube", "cp", "minikube:/tmp/data.tar.gz", f"{str(results_path)}/data.tar.gz"], logger=logger)
    run_command(["tar", "-xzf", f"{str(results_path)}/data.tar.gz", "-C", str(results_path)], logger=logger)

    logger.info(f"Resultados guardados en {results_path}")

    # Borrar carpeta temporal
    run_command(["minikube", "ssh", "--", "sudo", "rm", "-rf", "/tmp/data"], logger=logger)

    # Eliminar PV y PVC
    run_command(["minikube", "ssh", "--", "sudo", "rm", "-rf", KUBERNETES_PV_DIR], logger=logger)
    run_command(["kubectl", "delete", "pvc", pvc_name, "-n", namespace], logger=logger)
    run_command(["kubectl", "delete", "pv", pv_name], logger=logger)

    logger.info(f"PV ({pv_name}) y PVC ({pvc_name}) eliminados")


@flow
def main_flow():

    logger = get_run_logger()

    try:
        # Verificar prerrequisitos
        check_prerequisites()

        # Configurar simulación
        dia_actual = pd.to_datetime("2020-01-01")
        #dia_final = pd.to_datetime("2020-02-01")
        dia_final = pd.to_datetime("2020-01-02")

        args = parse_arguments()

        enviroment_dir = PROJECT_ROOT / "Entorno"

        # Preparar directorio para Docker y construir sus respectivas imagenes
        parent_dockerfile_dir = enviroment_dir / 'docker' / 'app'

        precio_flow_name, precio_dockerfile_dir = setup_dockerfile_dir(
            flow_script_path=args.precio.resolve(), 
            requirements_path=args.requirements.resolve(),
            data_path=args.data_precio.resolve(),
            base_image_name=args.base_image,
            parent_dockerfile_dir=parent_dockerfile_dir,
            additional_args=["--date", dia_actual.isoformat(), "--data", f"{DOCKER_DATA_DIR}{args.data_precio.resolve().name}"]
        )
        precio_image_name = build_docker_image(precio_dockerfile_dir, "precio")

        consumo_flow_name, consumo_dockerfile_dir = setup_dockerfile_dir(
            flow_script_path=args.consumo.resolve(), 
            requirements_path=args.requirements.resolve(),
            data_path=args.data_consumo.resolve(),
            base_image_name=args.base_image,
            parent_dockerfile_dir=parent_dockerfile_dir,
            additional_args=["--date", dia_actual.isoformat(), "--data", f"{DOCKER_DATA_DIR}{args.data_consumo.resolve().name}"]
        )
        consumo_image_name = build_docker_image(consumo_dockerfile_dir, "consumo")

        agente_flow_name, agente_dockerfile_dir = setup_dockerfile_dir(
            flow_script_path=args.agente.resolve(), 
            requirements_path=args.requirements.resolve(),
            base_image_name=args.base_image,
            parent_dockerfile_dir=parent_dockerfile_dir
        )
        agente_image_name = build_docker_image(agente_dockerfile_dir, "agente")
        
        # Preparar directorio para Kubernetes
        kubernetes_dir = enviroment_dir / 'kubernetes'

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

        # Simulación
        while dia_actual < dia_final:
            # Desplegar en Kubernetes
            deploy_to_kubernetes(
                flow_name=precio_flow_name,
                image_name=precio_image_name,
                configmap_name=configmap_name,
                secret_name=secret_name,
                pvc_name=pvc_name,
                kubernetes_dir=kubernetes_dir,
                namespace=args.namespace,
                timeout=args.timeout
            )

            deploy_to_kubernetes(
                flow_name=consumo_flow_name,
                image_name=consumo_image_name,
                configmap_name=configmap_name,
                secret_name=secret_name,
                pvc_name=pvc_name,
                kubernetes_dir=kubernetes_dir,
                namespace=args.namespace,
                timeout=args.timeout
            )

            deploy_to_kubernetes(
                flow_name=agente_flow_name,
                image_name=agente_image_name,
                configmap_name=configmap_name,
                secret_name=secret_name,
                pvc_name=pvc_name,
                kubernetes_dir=kubernetes_dir,
                namespace=args.namespace,
                timeout=args.timeout
            )

            dia_actual += timedelta(days=1)
        
        get_results(pv_name, pvc_name, namespace=args.namespace)

        return 0
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main_flow())
