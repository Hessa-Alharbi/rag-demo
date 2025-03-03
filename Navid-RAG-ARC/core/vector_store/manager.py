from typing import List, Dict, Any, Optional
from loguru import logger
from core.config import get_settings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import asyncio
import os

class VectorStoreManager:
    def __init__(self):
        self.settings = get_settings()
        self.store = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self.embeddings = None
        self.index_path = os.path.join(str(self.settings.VECTOR_STORE_DIR), "vector_index")

    async def initialize(self, namespace: str = "default"):
        """Initialize vector store with namespace support"""
        if self._initialized:
            return

        async with self._initialize_lock:
            if not self._initialized:
                try:
                    self.embeddings = self.settings.get_embeddings()
                    self.namespace = namespace
                    self.index_path = os.path.join(
                        str(self.settings.VECTOR_STORE_DIR), 
                        f"vector_index_{namespace}"
                    )
                    
                    os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                    
                    if os.path.exists(f"{self.index_path}.faiss"):
                        try:
                            self.store = FAISS.load_local(self.index_path, self.embeddings)
                            logger.info(f"Loaded existing vector store for namespace {namespace}")
                        except Exception as e:
                            logger.warning(f"Failed to load existing store, creating new one: {e}")
                            self.store = FAISS.from_texts(
                                texts=[""], 
                                embedding=self.embeddings
                            )
                    else:
                        self.store = FAISS.from_texts(
                            texts=[""], 
                            embedding=self.embeddings
                        )
                        self.store.save_local(self.index_path)
                        logger.info(f"Created new vector store for namespace {namespace}")
                    
                    self._initialized = True
                    
                except Exception as e:
                    logger.error(f"Failed to initialize vector store: {e}")
                    raise

    async def add_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Add documents to vector store"""
        if not self._initialized:
            await self.initialize()

        try:
            vector_ids = []
            for i, (text, metadata) in enumerate(zip(texts, metadatas or [{}] * len(texts))):
                vector_id = f"vec_{len(self.store.index_to_docstore_id)}_{i}"
                doc = Document(
                    page_content=text,
                    metadata={**metadata, "vector_id": vector_id}
                )
                self.store.add_documents([doc])
                vector_ids.append(vector_id)
            return vector_ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search with improved handling"""
        if not self._initialized:
            await self.initialize()

        try:
            if not query.strip():
                return []

            # Apply pre-processing to query
            cleaned_query = self._preprocess_query(query)
            
            # Perform search with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                docs = self.store.similarity_search(
                    cleaned_query,
                    k=k,
                    filter=filter
                )
            
            if not docs:
                logger.warning(f"No results found for query: {cleaned_query}")
                return []
                
            # Post-process and format results
            results = []
            for doc in docs:
                result = {
                    "id": doc.metadata.get("vector_id", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0.0)  # Add similarity score if available
                }
                results.append(result)
            
            return results

        except asyncio.TimeoutError:
            logger.error("Search operation timed out")
            return []
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def _preprocess_query(self, query: str) -> str:
        """Clean and prepare query for search"""
        # Remove extra whitespace
        query = " ".join(query.split())
        return query
