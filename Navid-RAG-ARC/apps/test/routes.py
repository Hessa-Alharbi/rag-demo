from fastapi import APIRouter, HTTPException
from apps.rag.services import RAGService
from core.llm.factory import ModelFactory
from core.config import get_settings
from loguru import logger
import asyncio
from functools import partial
import uuid
from core.vector_store.utils import create_milvus_collection
from core.vector_store.connection import MilvusConnectionManager
from pymilvus import utility

router = APIRouter(prefix="/test")

@router.post("/embeddings/test")
async def test_embeddings():
    """Test embeddings model"""
    try:
        settings = get_settings()
        embeddings = ModelFactory.create_embeddings()
        test_text = "This is a test document for embeddings."
        
        # Run sync method in threadpool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            embeddings.embed_query,
            test_text
        )
        
        return {
            "status": "success",
            "model": settings.EMBEDDING_MODEL,
            "embedding_size": len(result),
            "sample": result[:5]  # First 5 dimensions
        }
    except Exception as e:
        logger.error(f"Embeddings test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vector-store/test")
async def test_vector_store():
    """Test vector store operations"""
    settings = get_settings()
    test_collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    collection = None
    
    try:
        # Test collection creation
        logger.info(f"Creating test collection: {test_collection}")
        collection = create_milvus_collection(test_collection, settings.EMBEDDING_DIMENSION)
        
        # Test vector insertion
        test_data = [
            {
                "text": "This is a test document 1",
                "embeddings": [0.1] * settings.EMBEDDING_DIMENSION,
                "document_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                "chunk_index": 0
            },
            {
                "text": "This is a test document 2",
                "embeddings": [0.2] * settings.EMBEDDING_DIMENSION,
                "document_id": str(uuid.uuid4()),
                "conversation_id": str(uuid.uuid4()),
                "chunk_index": 1
            }
        ]
        
        # Insert data without IDs (using auto_id)
        mr = collection.insert(test_data)
        logger.info(f"Inserted {len(test_data)} test vectors")
        
        # Test search with simplified params
        results = collection.search(
            data=[[0.1] * settings.EMBEDDING_DIMENSION],
            anns_field="embeddings",
            param={"metric_type": settings.MILVUS_METRIC_TYPE},
            limit=2,
            output_fields=["text", "document_id"]
        )
        
        logger.info("Vector search test successful")
        return {"status": "success", "message": "Vector store tests passed"}
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        try:
            if collection and utility.has_collection(test_collection):
                utility.drop_collection(test_collection)
                logger.info(f"Cleaned up test collection: {test_collection}")
        except Exception as e:
            logger.warning(f"Failed to cleanup test collection: {e}")

@router.post("/llm/test")
async def test_llm():
    """Test LLM generation"""
    try:
        llm = ModelFactory.create_llm()
        test_prompt = "Write a short greeting in one sentence."
        
        # Ensure we're using agenerate for async operation
        result = await llm.agenerate([test_prompt])
        response = result.generations[0][0].text
        
        return {
            "status": "success",
            "model": llm.model_name,
            "response": response
        }
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hybrid-search/test")
async def test_hybrid_search():
    """Test hybrid search functionality"""
    try:
        rag_service = RAGService()
        
        # Test query
        query = "test document"
        conversation_id = "test-conversation"
        
        # Test search queries generation
        search_queries = await rag_service._generate_search_queries(query)
        
        # Test hybrid search
        results = await rag_service.hybrid_search(
            query=query,
            conversation_id=conversation_id,
            limit=2
        )
        
        return {
            "status": "success",
            "search_queries": search_queries,
            "search_results": results
        }
    except Exception as e:
        logger.error(f"Hybrid search test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
