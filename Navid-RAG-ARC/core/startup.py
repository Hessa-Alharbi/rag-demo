from core.vector_store.singleton import VectorStoreSingleton
from apps.rag.services import RAGService
from loguru import logger

async def initialize_vector_stores():
    """Initialize vector stores on application startup"""
    try:
        # Initialize global vector store
        await VectorStoreSingleton.get_instance()
        
        # Initialize RAG service
        rag_service = RAGService()
        await rag_service.initialize()
        
        logger.info("Vector stores initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector stores: {e}")
        raise
