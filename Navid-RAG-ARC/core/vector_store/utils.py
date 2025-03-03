from pymilvus import Collection, utility, connections, FieldSchema, CollectionSchema, DataType
from loguru import logger
from core.vector_store.connection import MilvusConnectionManager
from core.config import get_settings

def create_milvus_collection(collection_name: str, dimension: int):
    """Create Milvus collection with schema and index"""
    try:
        settings = get_settings()
        MilvusConnectionManager.ensure_connection()
        
        collection = None
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
        else:
            # Create new collection with schema
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=settings.MILVUS_AUTO_ID),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="conversation_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="chunk_index", dtype=DataType.INT64)
            ]
            schema = CollectionSchema(fields=fields, description="Document chunks collection")
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index with only metric_type for auto-indexing
            collection.create_index(
                field_name="embeddings",
                index_params={"metric_type": settings.MILVUS_METRIC_TYPE}
            )
            logger.info(f"Created collection with auto-index: {collection_name}")
        
        # Load collection into memory
        collection.load()
        return collection
            
    except Exception as e:
        logger.error(f"Error creating Milvus collection: {e}")
        raise
