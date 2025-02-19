import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / '.env' / '.env')

# DagsHub configuration
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN')
DAGSHUB_REPO_NAME = os.getenv('DAGSHUB_REPO_NAME')

# Data paths
DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'data'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', PROJECT_ROOT / 'models'))

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
