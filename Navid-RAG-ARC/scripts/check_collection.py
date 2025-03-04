import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from pymilvus import utility, Collection
from core.vector_store.connection import MilvusConnectionManager
from core.config import get_settings
from core.logger import logger

def check_collection():
    """Check the status of a Milvus collection and its indexes"""
    try:
        logger.info("Checking collection status...")
        settings = get_settings()
        
        # Ensure connection
        MilvusConnectionManager.ensure_connection()
        
        collection_name = settings.MILVUS_COLLECTION
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} does not exist.")
            return
        
        # Load collection
        collection = Collection(collection_name)
        collection.load()
        
        # Get collection stats
        stats = collection.stats()
        logger.info(f"Collection stats: {stats}")
        
        # Get collection schema
        schema = collection.schema
        logger.info(f"Collection schema: {schema}")
        
        # Get index information
        index_info = collection.index().params
        logger.info(f"Index information: {index_info}")
        
        # Check vector dimension
        for field in schema.fields:
            if field.dtype == 101:  # FLOAT_VECTOR type
                logger.info(f"Vector field: {field.name}, dimension: {field.params['dim']}")
        
    except Exception as e:
        logger.error(f"Error checking collection: {str(e)}")
        raise
    finally:
        MilvusConnectionManager.close_connection()

if __name__ == "__main__":
    try:
        check_collection()
        print("Collection check completed.")
    except Exception as e:
        print(f"Error during check: {str(e)}")
        sys.exit(1)
