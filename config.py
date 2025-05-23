from pathlib import Path
from dotenv import dotenv_values

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables from .env file
env_dict = dotenv_values(PROJECT_ROOT / '.env')

# DagsHub configuration
DAGSHUB_USERNAME = env_dict['DAGSHUB_USERNAME']
DAGSHUB_TOKEN = env_dict['DAGSHUB_TOKEN']
DAGSHUB_REPO_NAME = env_dict['DAGSHUB_REPO_NAME']

# Prefect Configuration
PREFECT_PROFILE = env_dict['PREFECT_PROFILE']
PREFECT_API_KEY = env_dict['PREFECT_API_KEY']
PREFECT_API_URL = env_dict['PREFECT_API_URL']

# Docker Configuration
DOCKER_DATA_DIR = env_dict['DOCKER_DATA_DIR']

# Kubernetes PVC configuration
KUBERNETES_PV_DIR = env_dict['KUBERNETES_PV_DIR']

# DagsHub Data Paths
REPO_DATA_DIR_PATH = env_dict['REPO_DATA_DIR_PATH']
REPO_MODELS_DIR_PATH = env_dict['REPO_MODELS_DIR_PATH']

__all__ = [
    "PROJECT_ROOT",
    "DAGSHUB_USERNAME",
    "DAGSHUB_REPO_NAME",
    "DAGSHUB_TOKEN",
    "PREFECT_PROFILE",
    "PREFECT_API_KEY",
    "PREFECT_API_URL",
    "DOCKER_DATA_DIR",
    "KUBERNETES_PV_DIR",
    "REPO_DATA_DIR_PATH",
    "REPO_MODELS_DIR_PATH"
]
