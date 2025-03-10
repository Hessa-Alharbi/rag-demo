"""
Advanced query processor for handling complex search queries and improving RAG relevance.
"""
from typing import Dict, List, Any, Optional, Tuple
import re
import asyncio
import json
from loguru import logger
from langdetect import detect
from core.llm.factory import ModelFactory
from core.llm.prompt_templates import COMPLEX_QUERY_PROCESSING_TEMPLATE
from core.language.arabic_utils import ArabicTextProcessor


class QueryProcessor:
    """
    Advanced query processor that handles complex queries with sophisticated
    preprocessing, query expansion, and entity recognition.
    """
    
    def __init__(self):
        """Initialize the query processor"""
        self.llm = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the query processor if not already initialized"""
        if not self._initialized:
            async with self._initialize_lock:
                if not self._initialized:
                    self.llm = ModelFactory.create_llm()
                    self._initialized = True
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query to enhance search quality.
        
        Args:
            query: The user's search query
            
        Returns:
            Dict with processed query components including:
            - original_query: The original query string
            - processed_query: The preprocessed query for search
            - expanded_queries: List of alternative query formulations
            - filters: Extracted metadata filters
            - concepts: Key concepts identified in the query
            - entities: Named entities identified in the query
            - query_type: The type of query (e.g., factoid, comparison, etc.)
            - language: The detected language of the query
        """
        if not query or not query.strip():
            return {
                "original_query": query,
                "processed_query": query,
                "expanded_queries": [],
                "filters": {},
                "concepts": [],
                "entities": [],
                "query_type": "unknown",
                "language": "unknown"
            }
            
        # Initialize
        await self.initialize()
        
        try:
            # Detect language
            language = "unknown"
            try:
                language = detect(query)
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
            
            # Basic preprocessing
            processed_query = self._basic_preprocessing(query)
            
            # For simple queries, do basic processing
            if len(query.split()) <= 3 or len(query) < 15:
                result = await self._process_simple_query(query, language)
            else:
                # For complex queries, use LLM-based processing
                result = await self._process_complex_query(query, language)
                
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            # Return basic fallback processing
            return {
                "original_query": query,
                "processed_query": self._basic_preprocessing(query),
                "expanded_queries": [query],
                "filters": {},
                "concepts": query.split(),
                "entities": [],
                "query_type": "unknown",
                "language": "unknown"
            }
            
    def _basic_preprocessing(self, query: str) -> str:
        """
        Perform basic query preprocessing:
        - Remove extra whitespace
        - Convert to lowercase
        - Remove certain punctuation
        
        Args:
            query: Original query string
            
        Returns:
            Clean, preprocessed query
        """
        # Remove extra whitespace
        clean_query = " ".join(query.split())
        
        # Remove redundant punctuation that doesn't affect meaning
        clean_query = re.sub(r'["\[\]{}\\|<>]', ' ', clean_query)
        
        # Remove extra spaces after cleaning
        clean_query = " ".join(clean_query.split())
        
        return clean_query
        
    async def _process_simple_query(self, query: str, language: str) -> Dict[str, Any]:
        """
        Process a simple query with basic techniques
        
        Args:
            query: The user query
            language: Detected language
            
        Returns:
            Processed query data
        """
        processed_query = self._basic_preprocessing(query)
        
        # Handle Arabic queries with specialized processing
        if language == 'ar' or ArabicTextProcessor.contains_arabic(query):
            arabic_data = ArabicTextProcessor.preprocess_arabic_query(query)
            expanded_queries = [arabic_data["normalized"]]
            expanded_queries.extend([kw for kw in arabic_data["keywords"] if kw])
            
            return {
                "original_query": query,
                "processed_query": arabic_data["normalized"],
                "expanded_queries": expanded_queries,
                "filters": {},
                "concepts": arabic_data["keywords"],
                "entities": [],
                "query_type": "simple",
                "language": "ar"
            }
        
        # For non-Arabic simple queries
        words = processed_query.split()
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'of', 'with', 'in', 'on', 'at', 'to', 'for'}
        keywords = [word for word in words if word.lower() not in stopwords]
        
        return {
            "original_query": query,
            "processed_query": processed_query,
            "expanded_queries": [processed_query],
            "filters": {},
            "concepts": keywords,
            "entities": self._extract_simple_entities(processed_query),
            "query_type": "simple",
            "language": language
        }
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """
        Extract potential named entities using simple heuristics
        
        Args:
            text: Input text
            
        Returns:
            List of potential entities
        """
        # Look for capitalized words as potential entities (English only)
        potential_entities = []
        
        # Simple regex for capitalized words
        cap_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
        if cap_words:
            potential_entities.extend(cap_words)
            
        # Look for quoted phrases that might be entities
        quoted = re.findall(r'"([^"]*)"', text)
        if quoted:
            potential_entities.extend(quoted)
            
        return potential_entities
        
    async def _process_complex_query(self, query: str, language: str) -> Dict[str, Any]:
        """
        Process a complex query using LLM-based analysis
        
        Args:
            query: The user query
            language: Detected language
            
        Returns:
            Processed query data with rich components
        """
        processed_query = self._basic_preprocessing(query)
        
        try:
            # Use LLM to analyze complex query
            async with asyncio.timeout(5):  # 5 second timeout
                prompt = COMPLEX_QUERY_PROCESSING_TEMPLATE.format(query=query)
                response = await self.llm.agenerate([prompt])
                response_text = response.generations[0][0].text
                
            # Extract JSON response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)
            
            # Clean up JSON text
            response_text = re.sub(r'^[^{]*', '', response_text)
            response_text = re.sub(r'[^}]*$', '', response_text)
            
            # Parse the JSON response
            query_analysis = json.loads(response_text)
            
            # Extract components with fallbacks
            concepts = query_analysis.get("concepts", [])
            if not concepts:
                concepts = processed_query.split()
                
            entities = query_analysis.get("entities", [])
            constraints = query_analysis.get("constraints", [])
            temporal = query_analysis.get("temporal_aspects", [])
            expanded_queries = query_analysis.get("expanded_queries", [processed_query])
            
            # Build filters from constraints
            filters = {}
            if constraints:
                for constraint in constraints:
                    # Try to extract key-value pairs from constraints
                    # Format like "year:2023" or "author:Smith"
                    if ":" in constraint:
                        key, value = constraint.split(":", 1)
                        filters[key.strip()] = value.strip()
            
            # Add temporal filters if present
            if temporal:
                filters["temporal"] = temporal
                
            return {
                "original_query": query,
                "processed_query": processed_query,
                "expanded_queries": expanded_queries,
                "filters": filters,
                "concepts": concepts,
                "entities": entities,
                "relationships": query_analysis.get("relationships", []),
                "primary_intent": query_analysis.get("primary_intent", ""),
                "query_type": "complex",
                "language": language
            }
                
        except (json.JSONDecodeError, ValueError, asyncio.TimeoutError, Exception) as e:
            logger.error(f"Complex query processing error: {e}")
            # Fallback to simple processing
            return await self._process_simple_query(query, language)
    
    async def generate_search_variations(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Generate search query variations to improve recall
        
        Args:
            query: Original query
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of query variations
        """
        await self.initialize()
        
        try:
            # Generate variations prompt
            prompt = f"""Generate {max_variations} alternative search queries for: "{query}"
            
            The variations should:
            1. Use different phrasings but preserve the same meaning
            2. Include both broader and narrower formulations
            3. Use synonyms where appropriate
            4. Keep the same language as the original query
            
            Return only the list of variations, one per line, with no additional text or numbering:"""
            
            async with asyncio.timeout(3):
                response = await self.llm.agenerate([prompt])
                variations_text = response.generations[0][0].text.strip()
                
            # Extract variations (one per line)
            variations = [v.strip() for v in variations_text.split("\n") if v.strip()]
            
            # Ensure we don't exceed max variations and don't include the original
            filtered_variations = [v for v in variations if v.lower() != query.lower()][:max_variations]
            
            # Add original query if no variations were generated
            if not filtered_variations:
                return [query]
                
            return filtered_variations
            
        except Exception as e:
            logger.warning(f"Error generating query variations: {e}")
            return [query]
    
    async def extract_query_facets(self, query: str) -> Dict[str, Any]:
        """
        Extract facets/attributes from the query for structured search
        
        Args:
            query: User query
            
        Returns:
            Dict of facets extracted from the query
        """
        # Extract attributes in format attribute:value or attribute=value
        facets = {}
        
        # Look for attribute:value or attribute=value patterns
        attr_patterns = re.finditer(r'(\w+)[:=]([^:=\s]+)', query)
        for match in attr_patterns:
            attr_name, attr_value = match.groups()
            facets[attr_name.lower()] = attr_value
        
        # Special handling for date ranges
        date_range = re.search(r'(?:from|between)\s+(\d{4}(?:-\d{2})?(?:-\d{2})?)\s+(?:to|and)\s+(\d{4}(?:-\d{2})?(?:-\d{2})?)', query)
        if date_range:
            start_date, end_date = date_range.groups()
            facets['date_range'] = {'start': start_date, 'end': end_date}
        
        # Handle negation patterns
        negation_patterns = re.finditer(r'not\s+(\w+)', query)
        excluded_terms = [match.group(1) for match in negation_patterns]
        if excluded_terms:
            facets['excluded'] = excluded_terms
            
        return facets