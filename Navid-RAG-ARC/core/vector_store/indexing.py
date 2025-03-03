from pymilvus import Collection, utility
from core.config import get_settings
from loguru import logger

async def create_collection_index(collection_name: str):
    """Create index for Milvus collection"""
    try:
        settings = get_settings()
        collection = Collection(collection_name)
        
        # Create IVF_FLAT index
        index_params = {
            "metric_type": settings.MILVUS_METRIC_TYPE,
            "index_type": settings.MILVUS_INDEX_TYPE,
            "params": settings.MILVUS_INDEX_PARAMS
        }
        
        collection.create_index(
            field_name="embeddings",
            index_params=index_params
        )
        
        # Load collection with index
        collection.load()
        logger.info(f"Created index for collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"Error creating collection index: {e}")
        raise
