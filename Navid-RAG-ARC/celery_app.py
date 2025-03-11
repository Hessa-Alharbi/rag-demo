"""
Main entry point for Celery workers
This file is used when starting Celery workers:
`celery -A celery_app worker --loglevel=info`
"""
import os
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Added project root to Python path: {project_root}")

try:
    # Import settings first to make sure environment is properly configured
    from core.settings import get_settings
    settings = get_settings()
    logger.info("Loaded settings")
    
    # Import the Celery instance
    from core.celery.config import celery_app
    logger.info("Loaded Celery configuration")
    
    # Configure imports manually rather than loading modules
    # This avoids potential circular imports and missing modules
    celery_app.conf.imports = [
        'core.celery.tasks.document_processing',
        'core.celery.tasks.embeddings',
    ]
    logger.info("Configured task imports")
    
    # Configure routes
    celery_app.conf.task_routes = {
        'process_document': {'queue': 'celery'},
        'generate_embeddings': {'queue': 'celery'},
        'generate_embeddings_async': {'queue': 'celery'},
        'core.celery.tasks.document_processing.process_document': {'queue': 'celery'},
        'core.celery.tasks.document_processing.generate_embeddings': {'queue': 'celery'},
        'core.celery.tasks.embeddings.generate_embeddings_async': {'queue': 'celery'},
    }
    logger.info("Configured task routes")
    
    # Make Celery app available at the module level
    app = celery_app
    
except Exception as e:
    logger.error(f"Error initializing Celery app: {e}")
    raise
    
logger.info("celery_app.py loaded successfully")

if __name__ == '__main__':
    app.start()
