from typing import List, Dict, Any, Optional, Tuple
from core.vector_store.manager import VectorStoreManager
from rank_bm25 import BM25Okapi
from loguru import logger
import numpy as np
from core.language.arabic_utils import ArabicTextProcessor
from sklearn.metrics.pairwise import cosine_similarity
import time

class HybridSearchIndex:
    def __init__(self, 
                vector_weight: float = 0.7,
                arabic_boost: float = 0.2,  # Boost Arabic results
                reranking_enabled: bool = True):
        self.vector_store = VectorStoreManager()
        self.bm25 = None
        self.documents = []
        self.vector_weight = vector_weight
        self.arabic_boost = arabic_boost
        self.reranking_enabled = reranking_enabled
        self.document_embeddings = {}  # Cache for document embeddings
        
    def index_chunks(self, chunks: List[Dict[str, Any]], doc_id: str, conversation_id: str):
        """Index chunks using both vector store and BM25 with enhanced metadata for better retrieval"""
        try:
            start_time = time.time()
            # Store documents for BM25
            self.documents.extend(chunks)
            
            # Create BM25 index with optimized content
            tokenized_chunks = []
            for chunk in chunks:
                # Use embedding_text if available, otherwise use content
                text_for_bm25 = chunk.get("embedding_text", chunk["content"])
                
                # For Arabic content, include both original and normalized forms for better matching
                if chunk.get("metadata", {}).get("is_arabic", False):
                    normalized = ArabicTextProcessor.normalize_arabic(text_for_bm25)
                    tokenized_chunks.append((normalized + " " + text_for_bm25).split())
                else:
                    tokenized_chunks.append(text_for_bm25.split())
                    
            if tokenized_chunks:
                self.bm25 = BM25Okapi(tokenized_chunks)
            
            # Add to vector store with enhanced metadata
            texts = []
            metadatas = []
            
            for chunk in chunks:
                # Use optimized embedding text if available
                embedding_text = chunk.get("embedding_text", chunk["content"])
                texts.append(embedding_text)
                
                # Enhanced metadata for better filtering and retrieval
                metadata = {
                    **chunk.get("metadata", {}),
                    "doc_id": doc_id,
                    "conversation_id": conversation_id,
                    "chunk_index": len(texts) - 1,  # Add chunk index for BM25 matching
                    "has_context": "context" in chunk and bool(chunk["context"]),
                    "content_type": "chunk"
                }
                
                # For Arabic content, add extra metadata
                if metadata.get("is_arabic", False):
                    metadata["arabic_keywords"] = chunk.get("metadata", {}).get("keywords", [])
                
                metadatas.append(metadata)
            
            # Index in vector store
            self.vector_store.add_texts(texts, metadatas)
            
            processing_time = time.time() - start_time
            logger.info(f"Indexed {len(texts)} chunks for document {doc_id} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise
            
    async def hybrid_search(
        self, 
        query: str,
        k: int = 5,
        filter_metadata: Dict[str, Any] = None,
        reranking_strategy: str = "hybrid"  # Options: "hybrid", "vector", "bm25"
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector similarity with enhanced reranking
        """
        start_time = time.time()
        try:
            # Check if query is Arabic for specialized processing
            is_arabic = ArabicTextProcessor.contains_arabic(query)
            
            # Process query based on language
            processed_query = query
            arabic_query_data = None
            
            if is_arabic:
                # Get enhanced query data for Arabic
                arabic_query_data = ArabicTextProcessor.preprocess_arabic_query(query)
                processed_query = arabic_query_data["normalized"]
                
                # Log Arabic query processing
                logger.debug(f"Arabic query detected. Original: '{query}', Normalized: '{processed_query}'")
                
            # Get vector search results with doubled k for reranking
            retrieval_k = k * 3 if self.reranking_enabled else k
            vector_results = await self.vector_store.similarity_search(
                query=processed_query,
                k=retrieval_k,
                filter=filter_metadata
            )
            
            # If we have very few results, try alternate query forms for Arabic
            if is_arabic and len(vector_results) < k and arabic_query_data:
                logger.debug(f"Few results ({len(vector_results)}), trying with keywords")
                
                # Try with keywords
                keyword_query = " ".join(arabic_query_data["keywords"])
                if keyword_query and keyword_query != processed_query:
                    keyword_results = await self.vector_store.similarity_search(
                        query=keyword_query,
                        k=retrieval_k,
                        filter=filter_metadata
                    )
                    
                    # Add unique results
                    existing_ids = {r.get("id") for r in vector_results if "id" in r}
                    vector_results.extend([r for r in keyword_results if r.get("id") not in existing_ids])
            
            # Skip BM25 if not initialized or not using hybrid strategy
            if not self.bm25 or not self.documents or reranking_strategy == "vector":
                logger.debug("Using vector search results only")
                vector_results = self._rerank_results(vector_results, query, is_arabic)
                return vector_results[:k]
                
            # Get BM25 scores for reranking
            if is_arabic and arabic_query_data:
                # Use both original and normalized for Arabic
                tokenized_query = processed_query.split()
                tokenized_query += arabic_query_data["stemmed"].split()
                tokenized_query = list(set(tokenized_query))  # Remove duplicates
            else:
                tokenized_query = query.split()
                
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores to [0-1] range
            if bm25_scores.size > 0 and bm25_scores.max() != bm25_scores.min():
                norm_bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
            else:
                norm_bm25_scores = np.zeros_like(bm25_scores) if bm25_scores.size > 0 else np.array([])
            
            # Apply hybrid scoring if we have vector results
            if vector_results:
                vector_scores = np.array([r.get("score", 0) for r in vector_results])
                
                # Normalize vector scores
                if vector_scores.size > 0 and vector_scores.max() != vector_scores.min():
                    norm_vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
                else:
                    norm_vector_scores = np.ones_like(vector_scores) if vector_scores.size > 0 else np.array([])
                
                # Dynamic weight adjustment for Arabic content
                alpha = self.vector_weight  # Base weight for vector scores
                
                # Combine scores (weighted average)
                for i, result in enumerate(vector_results):
                    doc_idx = result.get("metadata", {}).get("chunk_index", -1)
                    
                    if doc_idx >= 0 and doc_idx < len(norm_bm25_scores):
                        # Base hybrid score
                        combined_score = alpha * norm_vector_scores[i] + (1-alpha) * norm_bm25_scores[doc_idx]
                        
                        # Apply Arabic boosting if needed
                        if is_arabic and result.get("metadata", {}).get("is_arabic", False):
                            combined_score += self.arabic_boost
                        
                        result["score"] = float(combined_score)
                        
                        # Add score components for debugging
                        result["score_components"] = {
                            "vector": float(norm_vector_scores[i]) if i < len(norm_vector_scores) else 0,
                            "bm25": float(norm_bm25_scores[doc_idx]) if doc_idx < len(norm_bm25_scores) else 0,
                            "arabic_boost": self.arabic_boost if is_arabic and result.get("metadata", {}).get("is_arabic", False) else 0
                        }
                
                # Final reranking based on combined scores and other signals
                reranked = self._rerank_results(vector_results, query, is_arabic)
                search_time = time.time() - start_time
                logger.debug(f"Hybrid search completed in {search_time:.3f}s, returning {min(k, len(reranked))} results")
                return reranked[:k]
            
            # Fallback to vector results if we have no BM25 matches
            return vector_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Try to return vector results if available
            if 'vector_results' in locals() and vector_results:
                return vector_results[:k]
            return []
            
    def _rerank_results(self, results: List[Dict[str, Any]], query: str, is_arabic: bool = False) -> List[Dict[str, Any]]:
        """Apply additional reranking criteria beyond basic hybrid scores"""
        if not results:
            return []
            
        # Sort initially by score
        reranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        # Apply additional reranking for Arabic if needed
        if is_arabic:
            try:
                # For Arabic queries, use ArabicTextProcessor to enhance relevance
                reranked = ArabicTextProcessor.enhance_arabic_search_results(query, reranked)
            except Exception as e:
                logger.warning(f"Arabic reranking failed: {e}")
        
        # Boost results with context summaries
        for result in reranked:
            # Boost chunks that have context
            if result.get("metadata", {}).get("has_context", False):
                result["score"] = result.get("score", 0) * 1.05
                
            # Boost chunks that have matching topics
            topics = result.get("metadata", {}).get("topics", [])
            if topics and any(topic.lower() in query.lower() for topic in topics):
                result["score"] = result.get("score", 0) * 1.1
        
        # Final sort by score
        return sorted(reranked, key=lambda x: x.get("score", 0), reverse=True)
