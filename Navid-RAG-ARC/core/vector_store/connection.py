from pymilvus import connections
from core.config import get_settings
from loguru import logger

class MilvusConnectionManager:
    _is_connected = False

    @classmethod
    def ensure_connection(cls):
        """Ensure Milvus connection is established"""
        settings = get_settings()
        
        # Check if vector store provider is not Milvus
        if settings.VECTOR_STORE_PROVIDER.lower() != "milvus":
            logger.info(f"Using {settings.VECTOR_STORE_PROVIDER} instead of Milvus - skipping connection")
            cls._is_connected = True
            return
            
        # Only try to connect if Milvus is the configured provider
        if not cls._is_connected:
            try:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT
                )
                cls._is_connected = True
                logger.info("Successfully connected to Milvus")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}")
                # Don't raise exception to allow the application to continue with FAISS
                logger.warning("Continuing without Milvus connection - ensure vector store provider is set to FAISS in .env")

    @classmethod
    def close_connection(cls):
        """Close Milvus connection"""
        settings = get_settings()
        
        # Skip disconnection if we're not using Milvus
        if settings.VECTOR_STORE_PROVIDER.lower() != "milvus":
            return
            
        if cls._is_connected:
            try:
                connections.disconnect("default")
                cls._is_connected = False
                logger.info("Disconnected from Milvus")
            except Exception as e:
                logger.error(f"Error disconnecting from Milvus: {e}")
