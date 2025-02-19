import os
from prefect import flow, task
import mlflow
from mlflow.entities.view_type import ViewType
import dagshub
import dagshub.auth
from prefect.logging import get_logger

logger = get_logger()

@task(name="authenticate_dagshub", retries=2)
def authenticate_dagshub():
    """Task to authenticate with DagsHub using environment variables."""
    try:
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        if not dagshub_token:
            raise ValueError("DAGSHUB_TOKEN environment variable not set")
        
        dagshub.auth.add_app_token(token=dagshub_token)
        logger.info("Successfully authenticated with DagsHub")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with DagsHub: {str(e)}")
        raise

@task(name="initialize_dagshub")
def initialize_dagshub():
    """Task to initialize DagsHub connection."""
    try:
        username = os.getenv('DAGSHUB_USERNAME')
        repo_name = os.getenv('DAGSHUB_REPO_NAME')
        
        if not all([username, repo_name]):
            raise ValueError("Missing required environment variables: DAGSHUB_USERNAME or DAGSHUB_REPO_NAME")
        
        dagshub.init(
            repo_owner=username,
            repo_name=repo_name,
            mlflow=True
        )
        logger.info(f"Successfully initialized DagsHub for repo: {username}/{repo_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DagsHub: {str(e)}")
        raise

@task(name="test_mlflow_connection", retries=3)
def test_mlflow_connection():
    """Task to test MLflow connection by listing experiments."""
    try:
        experiments = mlflow.search_experiments(view_type=ViewType.ALL)
        logger.info("Successfully connected to MLflow!")
        logger.info(f"Found {len(experiments)} experiments")
        return len(experiments)
    except Exception as e:
        logger.error(f"Failed to connect to MLflow: {str(e)}")
        raise

@flow(name="mlflow_connection_test")
def mlflow_connection_flow():
    """Main flow to test MLflow connection through DagsHub."""
    try:
        # Execute tasks in sequence
        auth_success = authenticate_dagshub()
        if auth_success:
            init_success = initialize_dagshub()
            if init_success:
                num_experiments = test_mlflow_connection()
                logger.info(f"Flow completed successfully. Found {num_experiments} experiments.")
                return num_experiments
    except Exception as e:
        logger.error(f"Flow failed: {str(e)}")
        raise

if __name__ == "__main__":
    mlflow_connection_flow()