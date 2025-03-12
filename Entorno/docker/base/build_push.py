#!/usr/bin/env python3
"""
build_push.py

Script que construye una imagen de Docker con los recursos necesarios y la sube a un repositorio de Docker Hub.
Este script realiza las siguientes operaciones:
1. Copia el archivo requirements.txt al directorio del Dockerfile
2. Construye la imagen Docker con tags de fecha actual y 'latest'
3. Sube la imagen a Docker Hub

Uso:
    python build_push.py

Requisitos:
    - Docker instalado y configurado
    - Acceso a Docker Hub con credenciales válidas
    - Archivo requirements.txt en la raíz del proyecto
"""

import sys
import logging
from datetime import datetime
import subprocess
import shutil
import argparse
from typing import Tuple, Optional
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir la raíz del proyecto a sys.path
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
    from config import PROJECT_ROOT
except ImportError:
    logger.error("No se pudo importar la configuración del proyecto. Verificar la estructura del proyecto.")
    sys.exit(1)

def check_docker_installed() -> bool:
    """
    Verifica si Docker está instalado y disponible en el sistema.
    
    Returns:
        bool: True si Docker está disponible, False en caso contrario.
    """
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_docker_login() -> bool:
    """
    Verifica si hay una sesión activa en Docker Hub.
    
    Returns:
        bool: True si hay una sesión activa, False en caso contrario.
    """
    try:
        result = subprocess.run(
            ['docker', 'info'], 
            check=True, 
            capture_output=True, 
            text=True
        )
        return "Username:" in result.stdout
    except subprocess.SubprocessError:
        return False

def build_image(
    requirements_path: Path, 
    dockerfile_dir_path: Path,
    repository_name: str = "yagoutad/sgba1-base-image"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Copia el archivo de requisitos y construye la imagen Docker.
    
    Args:
        requirements_path: Ruta al archivo requirements.txt
        dockerfile_dir_path: Ruta al directorio que contiene el Dockerfile
        repository_name: Nombre del repositorio Docker donde se subirá la imagen
        
    Returns:
        Tuple[Optional[str], Optional[str]]: Tupla con el nombre y tag de la imagen creada,
        o (None, None) si hubo un error.
    """
    try:
        logger.info("Copiando archivo de requisitos...")
        # Copia el archivo de requirements.txt
        dest_requirements_path = dockerfile_dir_path / 'requirements.txt'
        
        if not requirements_path.exists():
            logger.error(f"No se encontró el archivo requirements.txt en {requirements_path}")
            return None, None
            
        shutil.copy2(requirements_path, dest_requirements_path)
        logger.info(f"Archivo requirements.txt copiado a {dest_requirements_path}")

        logger.info("Construyendo imagen Docker...")
        # Nombre y tag de la imagen
        image_name = repository_name
        image_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Construir la imagen
        build_cmd = [
            'docker', 'build', 
            '-t', f'{image_name}:{image_tag}', 
            '-t', f'{image_name}:latest', 
            str(dockerfile_dir_path)
        ]
        
        process = subprocess.run(
            build_cmd, 
            check=True, 
            capture_output=True,
            text=True
        )
        
        logger.info(f"Imagen Docker construida con éxito: {image_name}:{image_tag}")
        return image_name, image_tag
        
    except shutil.Error as e:
        logger.error(f"Error al copiar archivo requirements.txt: {e}")
        return None, None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al construir la imagen Docker: {e}")
        logger.error(f"Salida de error: {e.stderr}")
        return None, None
    except Exception as e:
        logger.error(f"Error inesperado durante la construcción de la imagen: {e}")
        return None, None

def push_image(image_name: str, image_tag: str) -> bool:
    """
    Sube la imagen Docker a Docker Hub.
    
    Args:
        image_name: Nombre de la imagen Docker
        image_tag: Tag de la imagen Docker
        
    Returns:
        bool: True si la imagen se subió correctamente, False en caso contrario.
    """
    try:
        logger.info(f"Subiendo imagen Docker {image_name}:{image_tag}...")
        
        # Subir la imagen con el tag específico
        push_cmd_tag = ['docker', 'push', f'{image_name}:{image_tag}']
        subprocess.run(push_cmd_tag, check=True, capture_output=True, text=True)
        logger.info(f"Imagen {image_name}:{image_tag} subida con éxito")
        
        # Subir la imagen con el tag 'latest'
        push_cmd_latest = ['docker', 'push', f'{image_name}:latest']
        subprocess.run(push_cmd_latest, check=True, capture_output=True, text=True)
        logger.info(f"Imagen {image_name}:latest subida con éxito")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al subir la imagen Docker: {e}")
        logger.error(f"Salida de error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado durante la subida de la imagen: {e}")
        return False

def parse_arguments():
    """
    Procesa los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Objeto con los argumentos procesados.
    """
    parser = argparse.ArgumentParser(
        description='Construye y sube una imagen Docker al repositorio'
    )
    parser.add_argument(
        '--requirements', 
        type=Path, 
        help='Ruta al archivo requirements.txt (por defecto: PROJECT_ROOT/requirements.txt)'
    )
    parser.add_argument(
        '--repository', 
        type=str, 
        default="yagoutad/sgba1-base-image",
        help='Nombre del repositorio Docker (por defecto: yagoutad/sgba1-base-image)'
    )
    parser.add_argument(
        '--only-build', 
        action='store_true',
        help='Solo construir la imagen, sin subirla'
    )
    parser.add_argument(
        '--verbose', 
        '-v', 
        action='store_true',
        help='Mostrar información detallada durante la ejecución'
    )
    
    return parser.parse_args()

def main() -> int:
    """
    Función principal que controla el flujo del script.
    
    Returns:
        int: Código de salida (0 si todo correcto, 1 si hay error)
    """
    args = parse_arguments()
    
    # Configurar nivel de log según verbosidad
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Verificar que Docker esté instalado
    if not check_docker_installed():
        logger.error("Docker no está instalado o no está disponible. Por favor, instale Docker para continuar.")
        return 1
        
    # Determinar rutas
    requirements_path = args.requirements if args.requirements else PROJECT_ROOT / 'requirements.txt'
    dockerfile_dir_path = Path(__file__).resolve().parent
    
    logger.info(f"Usando archivo requirements.txt: {requirements_path}")
    logger.info(f"Directorio del Dockerfile: {dockerfile_dir_path}")
    
    # Construir la imagen
    image_name, image_tag = build_image(
        requirements_path, 
        dockerfile_dir_path,
        repository_name=args.repository
    )
    
    if not image_name or not image_tag:
        logger.error("No se pudo construir la imagen Docker. Terminando ejecución.")
        return 1
    
    # Subir la imagen si se requiere
    if not args.only_build:
        # Verificar login en Docker Hub
        if not check_docker_login():
            logger.warning("No se detectó sesión activa en Docker Hub. Es posible que necesite ejecutar 'docker login' primero.")
            
        if not push_image(image_name, image_tag):
            logger.error("No se pudo subir la imagen Docker. Terminando ejecución.")
            return 1
    else:
        logger.info("Imagen Docker construida correctamente. Omitiendo subida.")
    
    logger.info("Proceso completado con éxito.")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario.")
        sys.exit(130)  # Código estándar para interrupción por SIGINT
    except Exception as e:
        logger.exception(f"Error no controlado: {e}")
        sys.exit(1)