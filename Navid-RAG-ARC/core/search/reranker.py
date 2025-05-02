from typing import List, Dict, Any, Optional
from loguru import logger
from core.llm.factory import ModelFactory
from core.llm.prompt_templates import RERANKING_PROMPT, QUERY_NORMALIZATION_TEMPLATE, CROSS_ENCODER_PROMPT
from collections import Counter
from core.language.arabic_utils import ArabicTextProcessor
from core.search.rank_fusion import RankFusion
from core.config import get_settings

class QueryResultReranker:
    """
    Advanced reranker that uses LLM reasoning, cross-encoding, and ensemble methods
    to improve search result relevance.
    """
    
    def __init__(self):
        """Initialize the reranker"""
        self.llm = None

    
    async def initialize(self):
        """Initialize the reranker with LLM model"""
        if self.llm is None:
            settings = get_settings()
            logger.info(f"Initializing reranker with LLM provider: {settings.LLM_PROVIDER}, model: {settings.LLM_MODEL}")
            self.llm = ModelFactory.create_llm()
            # Ensure we're using the configured model, not a hardcoded one
            if hasattr(self.llm, 'model_name'):
                logger.info(f"Reranker using model: {self.llm.model_name}")
            elif hasattr(self.llm, 'repo_id'):
                logger.info(f"Reranker using model: {self.llm.repo_id}")
    
    async def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        method: str = "ensemble"
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using the specified method.
        
        Args:
            query: User query
            results: List of search results to rerank
            top_k: Number of results to return after reranking
            method: Reranking method to use (keyword, semantic, cross_encoder, contextual, ensemble)
            
        Returns:
            Reranked list of results
        """
        if not results:
            return []
            
        if top_k is None:
            top_k = len(results)
            
        # Initialize LLM if needed
        await self.initialize()
        
        # Normalize query before reranking
        normalized_query = await self._normalize_query(query)
        
        try:
            # Select reranking method
            if method == "keyword":
                reranked = await self._keyword_rerank(normalized_query, results)
            elif method == "semantic":
                reranked = await self.semantic_rerank(normalized_query, results, top_k)
            elif method == "cross_encoder":
                reranked = await self.cross_encoder_rerank(normalized_query, results, top_k)
            elif method == "contextual":
                reranked = await self.contextual_rerank(normalized_query, results, top_k)
            elif method == "ensemble":
                reranked = await self.ensemble_rerank(normalized_query, results, top_k)
            else:
                # Default to keyword reranking as it's fastest
                reranked = await self._keyword_rerank(normalized_query, results)
                
            # Limit to top_k
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            # Fallback to original ordering
            return results[:top_k]
    
    async def _normalize_query(self, query: str) -> str:
        """
        Normalize the query by removing noise and extracting key terms.
        
        Args:
            query: Original user query
            
        Returns:
            Normalized query
        """
        # For Arabic queries, use specialized normalization
        if ArabicTextProcessor.contains_arabic(query):
            return ArabicTextProcessor.normalize_arabic(query)
            
        # For other queries, use LLM-based normalization
        try:
            prompt = QUERY_NORMALIZATION_TEMPLATE.format(query=query)
            response = await self.llm.agenerate([prompt])
            normalized = response.generations[0][0].text.strip()
            
            # If normalization produced something reasonable, use it
            if normalized and len(normalized) > 3:
                return normalized
                
            return query
        except Exception as e:
            logger.warning(f"Error normalizing query: {e}")
            return query
    
    async def _extract_query_concepts(self, query: str) -> List[str]:
        """
        Extract key concepts from query for better matching.
        
        Args:
            query: User query
            
        Returns:
            List of key concepts
        """
        # Handle Arabic
        if ArabicTextProcessor.contains_arabic(query):
            normalized = ArabicTextProcessor.normalize_arabic(query)
            # Remove Arabic stopwords
            words = normalized.split()
            concepts = [word for word in words if word and word not in ArabicTextProcessor.ARABIC_STOPWORDS]
            return concepts
        
        # For English/other languages, remove common stopwords
        stopwords = {
            'and', 'or', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'in',
            'for', 'to', 'of', 'with', 'by', 'as', 'but', 'if', 'from', 'it'
        }
        
        words = query.lower().split()
        concepts = [word for word in words if word and word not in stopwords]
        
        return concepts
    
    async def _keyword_rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fast keyword-based reranking using term frequency and n-gram matching.
        
        Args:
            query: User query
            results: Search results to rerank
            
        Returns:
            Reranked results
        """
        # Extract query terms and concepts
        query_terms = await self._extract_query_concepts(query)
        query_text = ' '.join(query_terms)
        
        # For each result, calculate a keyword score
        for result in results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Skip if no content
            if not content:
                result['keyword_score'] = 0
                continue
            
            # Calculate different relevance signals
            
            # 1. Term frequency score
            term_matches = 0
            content_lower = content.lower()
            for term in query_terms:
                term_count = content_lower.count(term.lower())
                term_matches += term_count
            
            # 2. Calculate n-gram matches (bi-grams and tri-grams)
            bigram_matches = self._count_ngram_matches(query_terms, content_lower, n=2)
            trigram_matches = self._count_ngram_matches(query_terms, content_lower, n=3) if len(query_terms) >= 3 else 0
            
            # 3. Exact phrase match (highest value)
            exact_match = 3.0 if query_text.lower() in content_lower else 0
            
            # 4. Title match (if title metadata exists)
            title_match = 0
            if 'title' in metadata:
                title = metadata['title'].lower()
                title_term_matches = sum(title.count(term.lower()) for term in query_terms)
                title_match = title_term_matches * 2  # Title matches worth more
            
            # Combine signals
            score = (
                term_matches * 1.0 +
                bigram_matches * 2.0 +
                trigram_matches * 3.0 +
                exact_match * 5.0 +
                title_match * 1.5
            )
            
            # Normalize score by content length (avoid bias towards very long documents)
            content_length = max(1, len(content) / 1000)  # length in thousands of chars
            normalized_score = score / (0.5 + 0.5 * content_length)
            
            result['keyword_score'] = normalized_score
        
        # Sort by keyword score
        reranked = sorted(results, key=lambda x: x.get('keyword_score', 0), reverse=True)
        
        return reranked
    
    def _count_ngram_matches(self, query_terms: List[str], content: str, n: int = 2) -> int:
        """
        Count n-gram matches between query terms and content.
        
        Args:
            query_terms: List of query terms
            content: Content to match against
            n: Size of n-grams to match
            
        Returns:
            Number of n-gram matches
        """
        if len(query_terms) < n:
            return 0
            
        # Generate n-grams from query terms
        query_ngrams = [' '.join(query_terms[i:i+n]) for i in range(len(query_terms)-n+1)]
        
        # Count matches
        match_count = sum(content.count(ngram.lower()) for ngram in query_ngrams)
        return match_count
    
    async def semantic_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to semantically rerank results based on query relevance.
        
        Args:
            query: User query
            results: Search results to rerank
            top_k: Number of results to return
            
        Returns:
            Semantically reranked results
        """
        if not results:
            return []
            
        # Prepare document content for reranking prompt
        document_texts = []
        for i, result in enumerate(results[:min(10, len(results))]):  # Limit to 10 docs for LLM
            content = result.get('content', '')
            if content:
                # Truncate long content
                if len(content) > 1000:
                    content = content[:997] + "..."
                document_texts.append(f"Document {i+1}:\n{content}")
        
        # Format the input for the reranking prompt
        documents_text = "\n\n".join(document_texts)
        prompt = RERANKING_PROMPT.format(query=query, documents=documents_text)
        
        try:
            # Generate reranking using LLM
            response = await self.llm.agenerate([prompt])
            response_text = response.generations[0][0].text.strip()
            
            # Try to parse the JSON response
            import re
            import json
            
            # Extract the JSON array from the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text.replace('\n', ' '), re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                rerank_data = json.loads(json_str)
                
                # Apply the reranking scores
                for item in rerank_data:
                    doc_id = item.get('document_id')
                    if doc_id is not None and 1 <= doc_id <= len(results):
                        results[doc_id-1]['semantic_score'] = item.get('score', 0)
                
                # Sort by semantic score
                reranked = sorted(results, key=lambda x: x.get('semantic_score', 0), reverse=True)
                
                return reranked[:top_k]
            else:
                logger.warning("Couldn't parse reranking response as JSON")
                # Fall back to keyword reranking
                return (await self._keyword_rerank(query, results))[:top_k]
                
        except Exception as e:
            logger.error(f"Error in semantic reranking: {e}")
            # Fall back to keyword reranking
            return (await self._keyword_rerank(query, results))[:top_k]
    
    async def cross_encoder_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Use cross-encoder style query-document pair scoring for more accurate ranking.
        
        Args:
            query: User query
            results: Search results to rerank
            top_k: Number of results to return
            
        Returns:
            Reranked results using cross-encoder scoring
        """
        if not results:
            return []
            
        # Limit to reasonable number for cross-encoder
        results_to_rerank = results[:min(15, len(results))]
        
        # Prepare query-document pairs for scoring
        pairs = []
        for i, result in enumerate(results_to_rerank):
            content = result.get('content', '')
            # Truncate long content
            if len(content) > 800:
                content = content[:797] + "..."
                
            pairs.append(f"Document {i}:\n{content}")
        
        pairs_text = "\n\n".join(pairs)
        
        # Format cross-encoder prompt
        prompt = CROSS_ENCODER_PROMPT.format(query=query, pairs=pairs_text)
        
        try:
            # Generate cross-encoder scores using LLM
            response = await self.llm.agenerate([prompt])
            response_text = response.generations[0][0].text.strip()
            
            # Try to parse the JSON output
            import re
            import json
            
            # Find JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text.replace('\n', ' '), re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                scores_data = json.loads(json_str)
                
                # Apply scores to results
                scored_results = []
                for item in scores_data:
                    doc_id = item.get('document_id')
                    if doc_id is not None and doc_id < len(results_to_rerank):
                        result = results_to_rerank[doc_id]
                        result['cross_encoder_score'] = item.get('score', 0)
                        scored_results.append(result)
                
                # Add any remaining results from the original list
                seen_ids = {r.get('id') for r in scored_results}
                for result in results:
                    if result.get('id') not in seen_ids:
                        result['cross_encoder_score'] = 0
                        scored_results.append(result)
                
                # Sort by cross-encoder score
                reranked = sorted(scored_results, key=lambda x: x.get('cross_encoder_score', 0), reverse=True)
                
                return reranked[:top_k]
                
            else:
                logger.warning("Couldn't parse cross-encoder response")
                return (await self._keyword_rerank(query, results))[:top_k]
                
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            return (await self._keyword_rerank(query, results))[:top_k]
    
    async def contextual_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank based on contextual relevance by analyzing key terms in both query and document.
        
        Args:
            query: User query
            results: Search results to rerank
            top_k: Number of results to return
            
        Returns:
            Contextually reranked results
        """
        if not results:
            return []
        
        # Extract key terms from query
        query_terms = self._extract_key_terms(query, max_terms=8)
        
        # For each result, calculate contextual relevance
        for result in results:
            content = result.get('content', '')
            if not content:
                result['contextual_score'] = 0
                continue
            
            # Extract key terms from content
            content_terms = self._extract_key_terms(content, max_terms=15)
            
            # Calculate semantic overlap
            overlap_score = 0
            
            # Count direct term overlap
            direct_matches = sum(1 for qt in query_terms if any(ct == qt for ct in content_terms))
            overlap_score += direct_matches * 2
            
            # Look for partial matches (substrings)
            partial_matches = sum(1 for qt in query_terms if any(qt in ct or ct in qt for ct in content_terms))
            overlap_score += partial_matches
            
            # Special handling for Arabic content
            if ArabicTextProcessor.contains_arabic(content) or ArabicTextProcessor.contains_arabic(query):
                for qt in query_terms:
                    for ct in content_terms:
                        # Check if terms are morphologically related
                        if ArabicTextProcessor.are_arabic_words_related(qt, ct):
                            overlap_score += 1.5
            
            result['contextual_score'] = overlap_score
        
        # Sort by contextual score
        reranked = sorted(results, key=lambda x: x.get('contextual_score', 0), reverse=True)
        
        return reranked[:top_k]
    
    def _extract_key_terms(self, text: str, max_terms: int = 10) -> List[str]:
        """
        Extract key terms from text for contextual matching.
        
        Args:
            text: Text to extract terms from
            max_terms: Maximum number of terms to extract
            
        Returns:
            List of key terms
        """
        # Check if text contains Arabic
        if ArabicTextProcessor.contains_arabic(text):
            return ArabicTextProcessor.extract_arabic_keywords(text, max_keywords=max_terms)
        
        # For other languages, use simpler keyword extraction
        # Remove stopwords
        stopwords = {
            'and', 'or', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'in', 'for',
            'to', 'of', 'with', 'by', 'as', 'but', 'if', 'from', 'it', 'this', 'that',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'can', 'could', 'should', 'would', 'might',
            'what', 'when', 'where', 'who', 'why', 'how'
        }
        
        # Tokenize and clean
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered_words = [word for word in words if word not in stopwords]
        
        # Count frequencies
        word_counts = Counter(filtered_words)
        
        # Get most common words
        common_words = [word for word, _ in word_counts.most_common(max_terms)]
        
        return common_words
    
    async def ensemble_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Ensemble multiple reranking methods for more robust results.
        
        Args:
            query: User query
            results: Search results to rerank
            top_k: Number of results to return
            
        Returns:
            Ensemble reranked results
        """
        if not results:
            return []
            
        if len(results) == 1:
            return results
            
        ranked_lists = []
        weights = []
        
        # 1. Keyword ranking (always include as it's fast)
        keyword_ranked = await self._keyword_rerank(query, results.copy())
        ranked_lists.append(keyword_ranked)
        weights.append(0.3)  # 30% weight
        
        # 2. Contextual ranking (fairly fast)
        contextual_ranked = await self.contextual_rerank(query, results.copy())
        ranked_lists.append(contextual_ranked)
        weights.append(0.3)  # 30% weight
        
        # 3. Use semantic or cross-encoder based on result count
        if len(results) <= 7:
            # For fewer results, use more accurate cross-encoder
            cross_encoder_ranked = await self.cross_encoder_rerank(query, results.copy())
            ranked_lists.append(cross_encoder_ranked)
            weights.append(0.4)  # 40% weight
        else:
            # For more results, use semantic ranking
            semantic_ranked = await self.semantic_rerank(query, results.copy())
            ranked_lists.append(semantic_ranked)
            weights.append(0.4)  # 40% weight
        
        # Apply special handling for Arabic
        if ArabicTextProcessor.contains_arabic(query):
            # Enhance with Arabic-specific reranking
            arabic_enhanced = ArabicTextProcessor.enhance_arabic_search_results(query, results.copy())
            ranked_lists.append(arabic_enhanced)
            weights.append(0.3)  # Give Arabic ranking 30% weight
            
            # Adjust other weights to normalize
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Use reciprocal rank fusion to combine rankings
        final_ranked = await self._reciprocal_rank_fusion(
            query=query,
            results=results,
            top_k=top_k,
            ranked_lists=ranked_lists,
            weights=weights
        )
        
        return final_ranked
    
    async def _reciprocal_rank_fusion(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        ranked_lists: Optional[List[List[Dict[str, Any]]]] = None,
        weights: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Implement reciprocal rank fusion to combine multiple rankings.
        
        Args:
            query: Original query
            results: Original results
            top_k: Number of results to return
            ranked_lists: List of different rankings to fuse
            weights: Weights for each ranking method
            
        Returns:
            Fused ranking of results
        """
        if not ranked_lists:
            return results[:top_k]
            
        # Use RankFusion utility
        fused_results = RankFusion.reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            weights=weights,
            id_key="id",
            k=60  # Standard RRF constant
        )
        
        # Add ensemble score
        for result in fused_results:
            result["ensemble_score"] = result.get("rrf_score", 0)
            
        return fused_results[:top_k]
