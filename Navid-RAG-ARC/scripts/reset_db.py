import os
import sys
import shutil
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from sqlmodel import SQLModel
from core.db import engine
from core.logger import logger

# Import all models to ensure they're registered with SQLModel
from apps.users.models import User
from apps.chat.models import Conversation, Message, Attachment
from apps.organizations.models import Organization, UserOrganization

UPLOAD_DIR = os.path.join(project_root, "uploads")

def clean_uploads_directory():
    """Remove and recreate the uploads directory"""
    try:
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR)
        logger.info("Uploads directory cleaned and recreated")
    except Exception as e:
        logger.error(f"Error cleaning uploads directory: {str(e)}")
        raise

def reset_database():
    """Drop all tables and recreate them"""
    try:
        logger.info("Starting database reset...")
        
        # Drop all tables
        logger.info("Dropping all tables...")
        SQLModel.metadata.drop_all(engine)
        
        # Recreate all tables
        logger.info("Creating new tables...")
        SQLModel.metadata.create_all(engine)
        
        # Clean uploads directory
        logger.info("Cleaning uploads directory...")
        clean_uploads_directory()
        
        logger.info("Database reset completed successfully!")
        
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Get confirmation from user
        response = input("WARNING: This will delete all data in the database and uploads directory. Are you sure? (y/N): ")
        if response.lower() == 'y':
            reset_database()
            print("Reset completed successfully!")
        else:
            print("Operation cancelled.")
    except Exception as e:
        print(f"Error during reset: {str(e)}")
        sys.exit(1)
