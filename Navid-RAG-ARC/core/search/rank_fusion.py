"""
Reciprocal Rank Fusion (RRF) and other advanced rank fusion methods
for combining multiple ranking strategies in RAG systems.
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import math

class RankFusion:
    """
    Advanced ranking fusion methods to combine multiple ranking strategies
    for more robust and accurate search results.
    """
    
    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: List[List[Dict[str, Any]]],
        weights: Optional[List[float]] = None,
        id_key: str = "id",
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Implements Reciprocal Rank Fusion (RRF) algorithm to combine multiple ranking lists.
        
        RRF score for a document d is: sum_i(1 / (k + r_i(d)))
        Where:
        - k is a constant (default 60)
        - r_i(d) is the rank of document d in the i-th ranking list
        
        Args:
            ranked_lists: List of lists of documents, each list already ranked by a different method
            weights: Optional weights for each ranking list, if None all lists are weighted equally
            id_key: Key to use as document identifier
            k: Constant in RRF formula, higher values decrease the impact of high ranks
            
        Returns:
            Combined and re-ranked list of documents
        """
        if not ranked_lists:
            return []
            
        # Initialize weights if not provided
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        else:
            # Normalize weights to sum to 1.0
            total = sum(weights)
            weights = [w / total for w in weights]
            
        # Map to store combined scores, using document ID as key
        doc_scores = defaultdict(float)
        
        # Map to store the original documents, using document ID as key
        docs_by_id = {}
        
        # Process each ranked list
        for i, ranked_list in enumerate(ranked_lists):
            # Get weight for this list
            weight = weights[i]
            
            # Process each document in the current ranked list
            for rank, doc in enumerate(ranked_list):
                # Get document ID, falling back to index as ID if not present
                doc_id = doc.get(id_key, f"doc_{i}_{rank}")
                
                # Store original document
                docs_by_id[doc_id] = doc
                
                # Compute RRF score component for this document in this list
                # Formula: weight * 1 / (k + rank)
                # Note: rank is 0-based, so we add 1 to make it 1-based for the formula
                rrf_score = weight * (1.0 / (k + rank + 1))
                
                # Add to document's cumulative score
                doc_scores[doc_id] += rrf_score
        
        # Sort documents by RRF score (descending)
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Create final ranked list
        final_ranked_list = []
        for doc_id in sorted_doc_ids:
            doc = docs_by_id[doc_id]
            # Add RRF score to document
            doc["rrf_score"] = doc_scores[doc_id]
            final_ranked_list.append(doc)
            
        return final_ranked_list
    
    @staticmethod
    def borda_count_fusion(
        ranked_lists: List[List[Dict[str, Any]]],
        weights: Optional[List[float]] = None,
        id_key: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        Implements weighted Borda Count fusion to combine multiple ranking lists.
        
        Borda score for a document d is: sum_i(weight_i * (N - rank_i(d)))
        Where:
        - N is the number of documents in the list
        - rank_i(d) is the rank of document d in the i-th ranking list
        
        Args:
            ranked_lists: List of lists of documents, each list already ranked by a different method
            weights: Optional weights for each ranking list, if None all lists are weighted equally
            id_key: Key to use as document identifier
            
        Returns:
            Combined and re-ranked list of documents
        """
        if not ranked_lists:
            return []
            
        # Initialize weights if not provided
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        else:
            # Normalize weights to sum to 1.0
            total = sum(weights)
            weights = [w / total for w in weights]
            
        # Map to store combined scores
        doc_scores = defaultdict(float)
        
        # Map to store documents
        docs_by_id = {}
        
        # Process each ranked list
        for i, ranked_list in enumerate(ranked_lists):
            # Get weight for this list
            weight = weights[i]
            
            # Get number of documents in this list
            n = len(ranked_list)
            
            # Process each document
            for rank, doc in enumerate(ranked_list):
                # Get document ID
                doc_id = doc.get(id_key, f"doc_{i}_{rank}")
                
                # Store document
                docs_by_id[doc_id] = doc
                
                # Compute Borda score for this document in this list
                # Higher ranks get higher scores: (n - rank)
                borda_score = weight * (n - rank)
                
                # Add to document's cumulative score
                doc_scores[doc_id] += borda_score
        
        # Sort documents by Borda score (descending)
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Create final ranked list
        final_ranked_list = []
        for doc_id in sorted_doc_ids:
            doc = docs_by_id[doc_id]
            # Add Borda score to document
            doc["borda_score"] = doc_scores[doc_id]
            final_ranked_list.append(doc)
            
        return final_ranked_list
    
    @staticmethod
    def combsum_fusion(
        scored_lists: List[List[Dict[str, Any]]],
        score_keys: List[str],
        weights: Optional[List[float]] = None,
        id_key: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        Implements CombSUM fusion which adds normalized scores across different rankings.
        
        Args:
            scored_lists: List of lists of documents, each with a score in a specified key
            score_keys: List of keys where scores are stored in each list
            weights: Optional weights for each score list
            id_key: Key to use as document identifier
            
        Returns:
            Combined and re-ranked list of documents
        """
        if not scored_lists or not score_keys or len(scored_lists) != len(score_keys):
            return []
            
        # Initialize weights if not provided
        if weights is None:
            weights = [1.0] * len(scored_lists)
        else:
            # Normalize weights to sum to 1.0
            total = sum(weights)
            weights = [w / total for w in weights]
            
        # First normalize scores in each list independently
        normalized_lists = []
        for i, (scored_list, score_key) in enumerate(zip(scored_lists, score_keys)):
            if not scored_list:
                normalized_lists.append([])
                continue
                
            # Find min and max score for normalization
            scores = [doc.get(score_key, 0) for doc in scored_list]
            if not scores:
                normalized_lists.append([])
                continue
                
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            # Normalize and store
            normalized_list = []
            for doc in scored_list:
                doc_copy = doc.copy()
                score = doc.get(score_key, 0)
                
                # Normalize to [0,1]
                if score_range > 0:
                    normalized_score = (score - min_score) / score_range
                else:
                    normalized_score = 1.0 if score > 0 else 0.0
                    
                # Store normalized score
                normalized_key = f"normalized_{score_key}"
                doc_copy[normalized_key] = normalized_score
                normalized_list.append(doc_copy)
                
            normalized_lists.append(normalized_list)
            
        # Map to store combined scores
        doc_scores = defaultdict(float)
        
        # Map to store documents
        docs_by_id = {}
        
        # Process each normalized list
        for i, (normalized_list, score_key) in enumerate(zip(normalized_lists, score_keys)):
            # Get weight for this list
            weight = weights[i]
            
            # Process each document
            for doc in normalized_list:
                # Get document ID
                doc_id = doc.get(id_key, f"doc_{i}")
                
                # Store document
                docs_by_id[doc_id] = doc
                
                # Get normalized score
                normalized_key = f"normalized_{score_key}"
                score = doc.get(normalized_key, 0) * weight
                
                # Add to document's cumulative score
                doc_scores[doc_id] += score
        
        # Sort documents by combined score (descending)
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        # Create final ranked list
        final_ranked_list = []
        for doc_id in sorted_doc_ids:
            doc = docs_by_id[doc_id]
            # Add combined score to document
            doc["combsum_score"] = doc_scores[doc_id]
            final_ranked_list.append(doc)
            
        return final_ranked_list
    
    @staticmethod
    def interleave_rankings(
        ranked_lists: List[List[Dict[str, Any]]],
        id_key: str = "id"
    ) -> List[Dict[str, Any]]:
        """
        Interleaves multiple ranking lists to improve diversity.
        Takes documents from each list in round-robin fashion.
        
        Args:
            ranked_lists: List of lists of documents, each already ranked
            id_key: Key to use as document identifier
            
        Returns:
            Interleaved list of documents
        """
        if not ranked_lists:
            return []
            
        # Remove empty lists
        ranked_lists = [lst for lst in ranked_lists if lst]
        
        # Set to track IDs we've already added
        seen_ids = set()
        
        # Result list
        result = []
        
        # Indexes into each list
        indexes = [0] * len(ranked_lists)
        
        # Round-robin selection
        while any(idx < len(lst) for idx, lst in zip(indexes, ranked_lists)):
            for i, ranked_list in enumerate(ranked_lists):
                # Skip if we've reached the end of this list
                if indexes[i] >= len(ranked_list):
                    continue
                    
                # Get next document from this list
                doc = ranked_list[indexes[i]]
                doc_id = doc.get(id_key, f"doc_{i}_{indexes[i]}")
                
                # Advance index for this list
                indexes[i] += 1
                
                # If we've already added this document, skip it
                if doc_id in seen_ids:
                    continue
                    
                # Add document to result
                result.append(doc)
                seen_ids.add(doc_id)
        
        return result
    
    @staticmethod
    def has_sufficient_diversity(
        doc_list: List[Dict[str, Any]],
        content_key: str = "content",
        threshold: float = 0.7,
        max_docs_to_check: int = 5
    ) -> bool:
        """
        Check if a list of documents has sufficient diversity.
        
        Args:
            doc_list: List of documents to check
            content_key: Key to access document content
            threshold: Similarity threshold below which documents are considered diverse
            max_docs_to_check: Maximum number of top documents to check for diversity
            
        Returns:
            True if documents are sufficiently diverse, False otherwise
        """
        if len(doc_list) <= 1:
            return True
            
        # Limit to top N documents
        docs_to_check = doc_list[:min(max_docs_to_check, len(doc_list))]
        
        # Extract content
        contents = [doc.get(content_key, "") for doc in docs_to_check]
        
        # Simple Jaccard similarity for text
        def jaccard_similarity(a: str, b: str) -> float:
            a_tokens = set(a.lower().split())
            b_tokens = set(b.lower().split())
            intersection = len(a_tokens.intersection(b_tokens))
            union = len(a_tokens.union(b_tokens))
            return intersection / union if union > 0 else 0.0
            
        # Check all pairs
        for i in range(len(contents)):
            for j in range(i+1, len(contents)):
                sim = jaccard_similarity(contents[i], contents[j])
                if sim > threshold:
                    # Found two similar documents
                    return False
        
        # All documents are sufficiently different
        return True