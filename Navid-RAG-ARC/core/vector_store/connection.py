from pymilvus import connections
from core.config import get_settings
from loguru import logger

class MilvusConnectionManager:
    _is_connected = False

    @classmethod
    def ensure_connection(cls):
        """Ensure Milvus connection is established"""
        if not cls._is_connected:
            settings = get_settings()
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
                raise

    @classmethod
    def close_connection(cls):
        """Close Milvus connection"""
        if cls._is_connected:
            try:
                connections.disconnect("default")
                cls._is_connected = False
                logger.info("Disconnected from Milvus")
            except Exception as e:
                logger.error(f"Error disconnecting from Milvus: {e}")
