from typing import List, Dict, Any, Optional
from loguru import logger
from core.config import get_settings
from langchain_community.vectorstores import FAISS
import asyncio
import os
from langchain_core.documents import Document as LCDocument

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
                                texts=["initialization text"], 
                                embedding=self.embeddings,
                                metadatas=[{"initialization": True}]
                            )
                    else:
                        self.store = FAISS.from_texts(
                            texts=["initialization text"], 
                            embedding=self.embeddings,
                            metadatas=[{"initialization": True}]
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
        """Add documents to vector store with improved metadata handling"""
        if not self._initialized:
            await self.initialize()

        try:
            if not texts:
                return []
                
            vector_ids = []
            documents = []
            
            for i, (text, metadata) in enumerate(zip(texts, metadatas or [{}] * len(texts))):
                if not text or not text.strip():
                    continue  # Skip empty texts
                    
                vector_id = f"vec_{self.namespace}_{len(self.store.index_to_docstore_id)}_{i}"
                doc_metadata = {**metadata, "vector_id": vector_id}
                
                # Create document with metadata
                doc = LCDocument(
                    page_content=text,
                    metadata=doc_metadata
                )
                documents.append(doc)
                vector_ids.append(vector_id)
            
            if documents:
                self.store.add_documents(documents)
                # Save after adding new documents
                self.store.save_local(self.index_path)
                
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
        """Perform similarity search with improved error handling and scoring"""
        if not self._initialized:
            await self.initialize()

        try:
            if not query or not query.strip():
                return []

            # Apply pre-processing to query
            cleaned_query = self._preprocess_query(query)
            
            # Perform search with timeout
            async with asyncio.timeout(10):  # 10 second timeout
                # Include similarity scores in search
                docs_and_scores = self.store.similarity_search_with_score(
                    cleaned_query,
                    k=k,
                    filter=filter
                )
            
            if not docs_and_scores:
                logger.warning(f"No results found for query: {cleaned_query}")
                return []
                
            # Post-process and format results
            results = []
            for doc, score in docs_and_scores:
                # Convert score to a similarity score (higher is better)
                similarity = 1.0 / (1.0 + score) if score > 0 else 1.0
                
                result = {
                    "id": doc.metadata.get("vector_id", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": similarity
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
        # Remove extra whitespace and normalize
        query = " ".join(query.split())
        
        # Check if query is in Arabic
        is_arabic = any(c for c in query if '\u0600' <= c <= '\u06FF')
        
        if is_arabic:
            # Import ArabicTextProcessor if not available
            from core.language.arabic_utils import ArabicTextProcessor
            # Use Arabic-specific normalization for Arabic queries
            return ArabicTextProcessor.normalize_arabic(query)
        
        # For non-Arabic queries, proceed with standard processing
        # Remove common question prefixes that might affect vector similarity
        prefixes = [
            "what is", "how do", "can you tell me", "please explain",
            "i want to know", "tell me about", "could you describe"
        ]
        
        lower_query = query.lower()
        for prefix in prefixes:
            if lower_query.startswith(prefix):
                query = query[len(prefix):].strip()
                break
                
        return query
