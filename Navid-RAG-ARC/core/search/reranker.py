from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio
import re
from core.llm.factory import ModelFactory
from core.llm.prompt_templates import RERANKING_PROMPT, QUERY_NORMALIZATION_TEMPLATE
import json

class QueryResultReranker:
    """
    Advanced reranker that uses LLM reasoning and keyword matching
    to improve search result relevance.
    """
    
    def __init__(self):
        """Initialize the reranker"""
        self.llm = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the reranker if not already initialized"""
        if not self._initialized:
            async with self._initialize_lock:
                if not self._initialized:
                    self.llm = ModelFactory.create_llm()
                    self._initialized = True
    
    async def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using advanced relevance scoring.
        
        Args:
            query: User query
            results: List of search results
            top_k: Number of top results to return (None for all)
            
        Returns:
            Reranked results
        """
        if not results:
            return []
            
        try:
            # Use simpler keyword-based reranking for better reliability
            reranked_results = await self._keyword_rerank(query, results)
            
            # Return top_k results if specified
            if top_k is not None:
                return reranked_results[:top_k]
                
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original results if reranking fails
            return results
    
    async def _normalize_query(self, query: str) -> str:
        """
        Normalize query using LLM to improve matching quality
        
        Args:
            query: User query
            
        Returns:
            Normalized query with key terms identified
        """
        try:
            # Initialize LLM if not already done
            await self.initialize()
            
            # Use a timeout to prevent long-running LLM calls
            async with asyncio.timeout(3):  # 3 second timeout for normalization
                # Use the query normalization template
                prompt = QUERY_NORMALIZATION_TEMPLATE.format(query=query)
                response = await self.llm.agenerate([prompt])
                normalized = response.generations[0][0].text.strip()
                
                # If response is too complex or appears to be an explanation, extract key terms
                if len(normalized) > len(query) * 2 or ":" in normalized:
                    lines = normalized.split('\n')
                    for line in lines:
                        if line and not line.startswith(('I', 'Here', 'The')):
                            return line.strip()
                    # Fallback to original query if can't extract a good line
                    return query
                
                return normalized
                
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Query normalization failed, using original query: {e}")
            return query  # Fallback to original query on error
    
    async def _keyword_rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using keyword matching and heuristics
        
        Args:
            query: User query
            results: List of search results
            
        Returns:
            Reranked results based on keyword matching
        """
        # Normalize query using LLM
        normalized_query = await self._normalize_query(query)
        query_terms = normalized_query.lower().split()
        
        # Detect if query is in Arabic
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in query)
        
        for result in results:
            # Get content
            content = result.get("content", "").lower()
            
            # Initialize score
            score = 0.0
            
            # 1. Term frequency
            term_count = sum(content.count(term) for term in query_terms if len(term) > 1)
            score += term_count * 0.5
            
            # 2. Exact phrase match (higher weight)
            if normalized_query.lower() in content:
                score += 3.0
            
            # 3. Title match (if title available in metadata)
            metadata = result.get("metadata", {})
            if "title" in metadata and query_terms:
                title = metadata["title"].lower()
                title_matches = sum(title.count(term) for term in query_terms if len(term) > 1)
                score += title_matches * 2.0  # Title matches have higher weight
                
            # 4. Position bias (earlier matches are better)
            for term in query_terms:
                if term and len(term) > 1 and term in content:
                    pos = content.find(term)
                    # Earlier positions get higher score
                    position_score = max(0, 1 - (pos / min(300, len(content))))
                    score += position_score * 0.5
                    
            # 5. Document length penalty (prefer shorter documents slightly)
            length_penalty = max(0, 1 - (len(content) / 2000))
            score += length_penalty * 0.2
            
            # 6. Arabic language handling (boost Arabic content for Arabic queries)
            if is_arabic:
                content_is_arabic = any('\u0600' <= c <= '\u06FF' for c in content)
                if content_is_arabic:
                    score += 1.5  # Boost Arabic content for Arabic queries
            
            # Store score in result
            result["relevance_score"] = score
            
        # Sort by score (descending)
        reranked = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        return reranked
    
    async def semantic_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using LLM-based semantic understanding.
        More accurate but slower than keyword reranking.
        
        Args:
            query: User query
            results: List of search results
            top_k: Number of top results to return
            
        Returns:
            Reranked results based on semantic relevance
        """
        if not results or len(results) <= 1:
            return results
            
        await self.initialize()
        
        try:
            # Format documents for LLM - limit to 10 to avoid token limits
            max_documents = min(10, len(results))
            docs_text = ""
            
            for i, result in enumerate(results[:max_documents]):
                content = result.get("content", "")
                # Truncate long documents to fit token limits
                if len(content) > 300:
                    content = content[:300] + "..."
                docs_text += f"Document {i+1}:\n{content}\n\n"
                
            # Create reranking prompt with correct parameter name
            prompt = RERANKING_PROMPT.format(
                query=query,
                documents=docs_text
            )
            
            # Call LLM with timeout
            async with asyncio.timeout(10):
                response = await self.llm.agenerate([prompt])
                response_text = response.generations[0][0].text
            
            # Try to parse JSON from response
            try:
                # Extract JSON if in markdown code block
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    response_text = json_match.group(1)
                
                # Handle potential text before or after JSON
                response_text = re.sub(r'^[^[{]*', '', response_text)
                response_text = re.sub(r'[^}\]]*$', '', response_text)
                    
                # Parse the JSON
                scores = json.loads(response_text)
                
                # Ensure scores is a list
                if isinstance(scores, dict):
                    scores = [scores]
                    
                # Apply scores to results
                id_to_score = {}
                for item in scores:
                    if isinstance(item, dict) and 'document_id' in item and 'score' in item:
                        try:
                            doc_id = int(item['document_id']) - 1  # Convert to 0-based index
                            if 0 <= doc_id < len(results):
                                id_to_score[doc_id] = float(item['score'])
                        except (ValueError, IndexError):
                            continue
                
                # Apply scores to original results
                for i, result in enumerate(results[:max_documents]):
                    if i in id_to_score:
                        result["semantic_score"] = id_to_score[i]
                    else:
                        # Assign fallback score for documents that didn't get scored
                        result["semantic_score"] = -1
                        
                # Sort by semantic score
                reranked = sorted(
                    results, 
                    key=lambda x: x.get("semantic_score", -1), 
                    reverse=True
                )
                
                # Return top k results
                return reranked[:top_k]
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing LLM reranking response: {str(e)}. Response: {response_text[:100]}...")
                # Fall back to keyword reranking
                return await self._keyword_rerank(query, results)[:top_k]
        
        except asyncio.TimeoutError:
            logger.warning("Semantic reranking timed out, falling back to keyword reranking")
            return await self._keyword_rerank(query, results)[:top_k]
        except Exception as e:
            logger.error(f"Error in semantic reranking: {str(e)}")
            return await self._keyword_rerank(query, results)[:top_k]
