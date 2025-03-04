"""Arabic language utilities for text processing and search optimization."""
import re
from typing import List, Dict, Any

# Arabic character ranges
ARABIC_CHARS = '\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'

# Arabic stopwords (common words that don't add much semantic meaning)
ARABIC_STOPWORDS = {
    'من', 'إلى', 'عن', 'على', 'في', 'هو', 'هي', 'هم', 'انت', 'انا', 'نحن', 'هذا', 'هذه', 'تلك',
    'ذلك', 'كان', 'كانت', 'مع', 'عند', 'لكن', 'و', 'ا', 'أن', 'إن', 'لم', 'لن', 'ثم', 'أو', 'ام',
    'اذا', 'ماذا', 'كيف', 'اين', 'متى', 'لماذا', 'كم', 'اي', 'فى', 'الى', 'الذي', 'التي'
}

# Diacritics and tatweel characters
ARABIC_DIACRITICS = '\u064B-\u065F\u0670'
TATWEEL = '\u0640'

class ArabicTextProcessor:
    """Utilities for processing Arabic text in search and indexing contexts."""
    
    @staticmethod
    def is_arabic(text: str) -> bool:
        """Check if text contains Arabic characters."""
        arabic_pattern = re.compile(f'[{ARABIC_CHARS}]')
        return bool(arabic_pattern.search(text))
    
    @staticmethod
    def contains_arabic(text: str) -> bool:
        """Check if text contains any Arabic characters."""
        return any(c for c in text if '\u0600' <= c <= '\u06FF')
    
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """
        Normalize Arabic text for better search matching:
        - Remove diacritics (tashkeel)
        - Normalize alef variants
        - Remove tatweel (kashida)
        - Normalize Arabic numerals
        """
        if not text:
            return text
            
        # Remove diacritics
        text = re.sub(f'[{ARABIC_DIACRITICS}]', '', text)
        
        # Remove tatweel (kashida)
        text = re.sub(f'[{TATWEEL}]', '', text)
        
        # Normalize alef variants
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        
        # Normalize ya and alef maksura
        text = text.replace('ى', 'ي').replace('ئ', 'ي')
        
        # Normalize teh marbuta to heh
        text = text.replace('ة', 'ه')
        
        # Normalize Arabic numerals to Latin
        digit_map = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
                    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'}
        for ar, en in digit_map.items():
            text = text.replace(ar, en)
            
        return text
    
    @staticmethod
    def remove_arabic_stopwords(text: str) -> str:
        """Remove Arabic stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word not in ARABIC_STOPWORDS]
        return ' '.join(filtered_words)
    
    @staticmethod
    def preprocess_arabic_query(query: str) -> Dict[str, Any]:
        """
        Preprocess an Arabic query for search optimization.
        Returns both normalized and keyword versions.
        """
        if not query or not ArabicTextProcessor.contains_arabic(query):
            return {"original": query, "normalized": query, "keywords": []}
            
        # Normalize the query
        normalized = ArabicTextProcessor.normalize_arabic(query)
        
        # Extract keywords (remove stopwords and split)
        words = normalized.split()
        keywords = [w for w in words if w and w not in ARABIC_STOPWORDS]
        
        return {
            "original": query,
            "normalized": normalized,
            "keywords": keywords
        }
    
    @staticmethod
    def enhance_arabic_search_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance search results for Arabic queries by:
        - Boosting results with exact Arabic matches
        - Promoting results with similar Arabic terms
        """
        if not results or not ArabicTextProcessor.contains_arabic(query):
            return results
            
        # Normalize the query for comparison
        normalized_query = ArabicTextProcessor.normalize_arabic(query)
        query_terms = normalized_query.split()
        
        for result in results:
            # Get content
            content = result.get("content", "")
            
            # Skip if no content
            if not content:
                continue
                
            # Process content if it contains Arabic
            if ArabicTextProcessor.contains_arabic(content):
                # Normalize content
                normalized_content = ArabicTextProcessor.normalize_arabic(content.lower())
                
                # Arabic exact match bonus
                if normalized_query in normalized_content:
                    result["score"] = result.get("score", 0) + 2.0
                
                # Arabic term match bonus
                term_matches = sum(normalized_content.count(term) for term in query_terms if len(term) > 1)
                result["score"] = result.get("score", 0) + (term_matches * 0.5)
        
        # Sort by updated scores
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
    @staticmethod
    def extract_arabic_keywords(text: str) -> List[str]:
        """Extract meaningful Arabic keywords from text."""
        if not text:
            return []
            
        # Normalize the text
        normalized = ArabicTextProcessor.normalize_arabic(text)
        
        # Split into words
        words = re.findall(f'[{ARABIC_CHARS}]+', normalized)
        
        # Filter out stopwords and short words
        keywords = [w for w in words if w not in ARABIC_STOPWORDS and len(w) > 1]
        
        return keywords
