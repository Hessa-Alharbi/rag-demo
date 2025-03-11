from celery import Celery
import sys
import os
import logging
from core.settings import get_settings

settings = get_settings()

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Initializing Celery configuration")

# Create Celery app
celery_app = Celery(
    "navid_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Fix for Windows-specific permission issues
if sys.platform.startswith('win'):
    # Use 'solo' pool on Windows to avoid permission issues
    celery_app.conf.worker_pool = "solo"
    logger.info("Using 'solo' pool for Windows")

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json", "application/json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_max_tasks_per_child=100,  # Restart worker after processing N tasks
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit for tasks
    worker_cancel_long_running_tasks_on_connection_loss=True,
    result_accept_content=["json", "application/json"],
    task_remote_tracebacks=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

# Set autodiscover_tasks to False to avoid automatic module loading
# We'll manually specify imports in celery_app.py
celery_app.autodiscover_tasks = lambda: []

# Clear Redis result backend from any corrupted data on startup
try:
    from redis import Redis
    redis = Redis.from_url(settings.CELERY_RESULT_BACKEND)
    
    keys = redis.keys('celery-task-meta-*')
    if keys:
        logger.info(f"Found {len(keys)} task results in Redis")
        # Only clear old failed tasks if needed
        # This is safer than clearing everything
        # redis.delete(*keys)
except Exception as e:
    logger.error(f"Could not connect to Redis result backend: {e}")

logger.info("Celery configuration complete")
