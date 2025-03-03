from loguru import logger
import sys
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Remove default logger
logger.remove()

# Add console logger with custom format
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file logger with rotation and retention
log_file = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(
    log_file,
    rotation="12:00",  # New file at midnight
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    backtrace=True,
    diagnose=True
)

# Add error log file
error_log_file = "logs/error.log"
logger.add(
    error_log_file,
    rotation="100 MB",
    retention="60 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    level="ERROR",
    backtrace=True,
    diagnose=True
)

def setup_logging(app_name: str = "navid-rag"):
    """Configure additional logging parameters"""
    logger.configure(extra={"app_name": app_name})
