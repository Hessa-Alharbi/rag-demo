import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from pymilvus import utility
from core.vector_store.connection import MilvusConnectionManager
from core.vector_store.utils import create_milvus_collection
from core.config import get_settings
from core.logger import logger

def reset_vector_store():
    """Reset Milvus collections by dropping and recreating them"""
    try:
        logger.info("Starting vector store reset...")
        settings = get_settings()
        
        # Ensure connection
        MilvusConnectionManager.ensure_connection()
        
        collection_name = settings.MILVUS_COLLECTION
        
        # Drop collection if exists
        if utility.has_collection(collection_name):
            logger.info(f"Dropping collection: {collection_name}")
            utility.drop_collection(collection_name)
        
        # Create new collection
        logger.info(f"Creating new collection: {collection_name}")
        collection = create_milvus_collection(collection_name, settings.EMBEDDING_DIMENSION)
        
        # Create index
        logger.info("Creating index...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(
            field_name="embeddings",
            index_params=index_params
        )
        logger.info("Vector store reset completed successfully!")
        
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        raise
    finally:
        MilvusConnectionManager.close_connection()

if __name__ == "__main__":
    try:
        response = input("WARNING: This will delete all vectors in Milvus. Are you sure? (y/N): ")
        if response.lower() == 'y':
            reset_vector_store()
            print("Reset completed successfully!")
        else:
            print("Operation cancelled.")
    except Exception as e:
        print(f"Error during reset: {str(e)}")
        sys.exit(1)
