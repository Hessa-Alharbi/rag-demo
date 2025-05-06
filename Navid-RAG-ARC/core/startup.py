import os
from loguru import logger
from core.config import get_settings
from core.vector_store.connection import MilvusConnectionManager
from core.llm.factory import ModelFactory

def create_embedding_model():
    """مؤقتًا: إنشاء نموذج التضمين المستخدم في النظام."""
    settings = get_settings()
    logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL}")
    embeddings = ModelFactory.create_embeddings()
    logger.info(f"Embedding model initialized successfully.")
    return embeddings

def initialize_vector_stores():
    """Initialize and ensure vector stores are ready."""
    settings = get_settings()
    logger.info("Initializing vector stores...")
    
    # Ensure Milvus connection
    MilvusConnectionManager.ensure_connection()
    
    # Initialize embedding model for indexing
    create_embedding_model()
    
    logger.info("Vector stores initialized successfully")

def cleanup():
    """Clean up resources before shutdown."""
    logger.info("Cleaning up resources...")
    
    # Close Milvus connections
    try:
        MilvusConnectionManager.disconnect_all()
        logger.info("Milvus connections closed")
    except Exception as e:
        logger.error(f"Error closing Milvus connections: {str(e)}")
    
    logger.info("Cleanup completed")
