"""Arabic language utilities for text processing and search optimization."""
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import string

# Arabic character ranges
ARABIC_CHARS = '\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF'

# Extended Arabic stopwords (comprehensive list for better filtering)
ARABIC_STOPWORDS = {
    'من', 'إلى', 'عن', 'على', 'في', 'هو', 'هي', 'هم', 'انت', 'انا', 'نحن', 'هذا', 'هذه', 'تلك',
    'ذلك', 'كان', 'كانت', 'مع', 'عند', 'لكن', 'و', 'ا', 'أن', 'إن', 'لم', 'لن', 'ثم', 'أو', 'ام',
    'اذا', 'ماذا', 'كيف', 'اين', 'متى', 'لماذا', 'كم', 'اي', 'فى', 'الى', 'الذي', 'التي', 'اللذين',
    'اللتين', 'الذين', 'اللاتي', 'الذي', 'التي', 'لكن', 'ليت', 'لعل', 'كأن', 'وكأن', 'حتى', 'إذا',
    'إن', 'أن', 'لن', 'لم', 'ثم', 'سوف', 'قد', 'كي', 'منذ', 'مذ', 'عدا', 'خلا', 'حاشا', 'إما', 'لا',
    'ولا', 'ما', 'لات', 'كلا', 'ذا', 'ولذا', 'هيا', 'آه', 'آها', 'وا', 'آي', 'آ', 'أي', 'فيما',
    'ولقد', 'فقد', 'لقد', 'بل', 'مما', 'إما', 'إلا', 'ولا', 'لما', 'بما', 'هلا', 'ألا', 'ولما',
    'لولا', 'وإن', 'فلا', 'وإلا', 'إذ', 'إذا', 'حيث', 'كلما', 'فإن', 'لو', 'أما', 'إما', 'فإما',
    'وأما', 'ومن', 'فمن', 'وما', 'فما', 'وإنما', 'ففي', 'وفي', 'ومع', 'فمع', 'وعن', 'فعن', 'وعلى',
    'فعلى', 'وإلى', 'فإلى', 'كما'
}

# Define word prefixes in Arabic that could be removed for stemming
ARABIC_PREFIXES = ['ال', 'بال', 'كال', 'فال', 'لل', 'وال']

# Define word suffixes in Arabic that could be removed for stemming
ARABIC_SUFFIXES = ['ه', 'ها', 'ك', 'ي', 'هم', 'هن', 'كم', 'كن', 'نا', 'ون', 'ين', 'ان', 'تين', 'تان']

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
        
        # Normalize hamza forms
        text = text.replace('ؤ', 'و')
        
        # Normalize teh marbuta to heh
        text = text.replace('ة', 'ه')
        
        # Normalize Arabic numerals to Latin
        digit_map = {'٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
                    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'}
        for ar, en in digit_map.items():
            text = text.replace(ar, en)
            
        # Remove punctuation and special characters
        arabic_punctuation = '،؛؟»«""٪'
        translator = str.maketrans('', '', arabic_punctuation + string.punctuation)
        text = text.translate(translator)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
            
        return text
    
    @staticmethod
    def remove_arabic_stopwords(text: str) -> str:
        """Remove Arabic stopwords from text."""
        words = text.split()
        filtered_words = [word for word in words if word not in ARABIC_STOPWORDS]
        return ' '.join(filtered_words)
    
    @staticmethod
    def light_stem_arabic(word: str) -> str:
        """
        Apply light stemming to Arabic words by removing common prefixes and suffixes.
        This is a simplified approach compared to full stemming algorithms.
        """
        if len(word) <= 3:  # Don't stem very short words
            return word
            
        # Try to remove prefixes (only one, from longest to shortest)
        for prefix in sorted(ARABIC_PREFIXES, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                word = word[len(prefix):]
                break
                
        # Try to remove suffixes (only one, from longest to shortest)
        for suffix in sorted(ARABIC_SUFFIXES, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                break
                
        return word
    
    @staticmethod
    def apply_stemming(text: str) -> str:
        """Apply light stemming to all words in a text."""
        words = text.split()
        stemmed_words = [ArabicTextProcessor.light_stem_arabic(word) for word in words]
        return ' '.join(stemmed_words)
    
    @staticmethod
    def preprocess_arabic_query(query: str) -> Dict[str, Any]:
        """
        Preprocess an Arabic query for search optimization.
        Returns normalized, stemmed, and keyword versions with various expansions.
        """
        if not query or not ArabicTextProcessor.contains_arabic(query):
            return {"original": query, "normalized": query, "keywords": [], "stemmed": query, "expansions": []}
            
        # Normalize the query
        normalized = ArabicTextProcessor.normalize_arabic(query)
        
        # Apply stemming
        stemmed = ArabicTextProcessor.apply_stemming(normalized)
        
        # Extract keywords (remove stopwords and split)
        words = normalized.split()
        keywords = [w for w in words if w and w not in ARABIC_STOPWORDS]
        
        # Generate query expansions (combinations of original, normalized, and stemmed)
        expansions = [query, normalized, stemmed]
        
        # Add stemmed keywords
        stemmed_keywords = [ArabicTextProcessor.light_stem_arabic(w) for w in keywords]
        expansions.extend(stemmed_keywords)
        
        # Add bigrams if available
        if len(keywords) >= 2:
            bigrams = [' '.join(keywords[i:i+2]) for i in range(len(keywords)-1)]
            expansions.extend(bigrams[:2])  # Add only first two bigrams
        
        # Remove duplicates while preserving order
        unique_expansions = []
        seen = set()
        for exp in expansions:
            if exp not in seen and exp.strip():
                unique_expansions.append(exp)
                seen.add(exp)
        
        return {
            "original": query,
            "normalized": normalized,
            "keywords": keywords,
            "stemmed": stemmed,
            "stemmed_keywords": stemmed_keywords,
            "expansions": unique_expansions
        }
    
    @staticmethod
    def enhance_arabic_search_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance search results for Arabic queries using advanced linguistic techniques:
        - Boosting results with exact Arabic matches
        - Promoting results with stemmed term matches
        - Boosting based on term proximity in Arabic content
        - Considering morphological variations
        """
        if not results or not ArabicTextProcessor.contains_arabic(query):
            return results
            
        # Process query with full preprocessing
        query_data = ArabicTextProcessor.preprocess_arabic_query(query)
        normalized_query = query_data["normalized"]
        stemmed_query = query_data["stemmed"]
        query_terms = normalized_query.split()
        stemmed_terms = query_data["stemmed_keywords"]
        
        for result in results:
            # Get content
            content = result.get("content", "")
            
            # Skip if no content
            if not content:
                continue
                
            # Process content if it contains Arabic
            if ArabicTextProcessor.contains_arabic(content):
                # Apply full preprocessing to content
                normalized_content = ArabicTextProcessor.normalize_arabic(content.lower())
                stemmed_content = ArabicTextProcessor.apply_stemming(normalized_content)
                
                # Initialize score components
                exact_match_score = 0
                term_match_score = 0
                stemmed_match_score = 0
                proximity_score = 0
                
                # 1. Exact phrase match (highest value)
                if normalized_query in normalized_content:
                    exact_match_score = 3.0
                
                # 2. Individual term matches
                term_matches = sum(normalized_content.count(term) for term in query_terms if len(term) > 1)
                term_match_score = term_matches * 0.5
                
                # 3. Stemmed term matches (helps catch morphological variations)
                stemmed_matches = sum(stemmed_content.count(term) for term in stemmed_terms if len(term) > 1)
                stemmed_match_score = stemmed_matches * 0.7
                
                # 4. Term proximity (check if terms appear close to each other)
                if len(query_terms) > 1:
                    content_words = normalized_content.split()
                    for i, word in enumerate(content_words):
                        # Check if current word matches any query term
                        if any(term == word for term in query_terms if len(term) > 1):
                            # Check next few words for other query terms
                            window = content_words[i:i+5]
                            for term in query_terms:
                                if term != word and term in window:
                                    proximity_score += 0.5
                
                # 5. Position bias (terms appearing earlier in document)
                position_score = 0
                for term in query_terms:
                    if term in normalized_content:
                        pos = normalized_content.find(term)
                        # Higher score for earlier positions
                        position_score += max(0, 0.3 - (pos / min(500, len(normalized_content))))
                
                # Calculate final score with weighted components
                final_score = (
                    exact_match_score * 1.0 +
                    term_match_score * 0.7 +
                    stemmed_match_score * 0.8 +
                    proximity_score * 0.6 +
                    position_score * 0.5
                )
                
                # Update result score
                result["arabic_score"] = final_score
                result["score"] = result.get("score", 0) + final_score
                
                # Add detailed scoring components for debugging
                result["score_details"] = {
                    "exact_match": exact_match_score,
                    "term_match": term_match_score,
                    "stemmed_match": stemmed_match_score,
                    "proximity": proximity_score,
                    "position": position_score
                }
        
        # Sort by updated scores
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    
    @staticmethod
    def extract_arabic_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract meaningful Arabic keywords from text using TF-IDF-like approach."""
        if not text:
            return []
            
        # Normalize the text
        normalized = ArabicTextProcessor.normalize_arabic(text)
        
        # Split into words and filter stopwords
        words = normalized.split()
        filtered_words = [w for w in words if w not in ARABIC_STOPWORDS and len(w) > 1]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Get most common terms but prefer longer words when counts are similar
        scored_words = [(word, count * (0.5 + 0.1 * min(len(word), 10))) 
                        for word, count in word_counts.items()]
        
        # Sort by adjusted score
        sorted_words = sorted(scored_words, key=lambda x: x[1], reverse=True)
        
        # Return top keywords
        return [word for word, _ in sorted_words[:max_keywords]]
    
    @staticmethod
    def segment_arabic_text(text: str) -> List[str]:
        """
        Segment Arabic text into semantically meaningful chunks.
        Tries to break at sentence boundaries while preserving context.
        """
        if not text:
            return []
            
        # Clean and normalize
        text = ArabicTextProcessor.normalize_arabic(text)
        
        # First try to split by Arabic full stop
        sentences = re.split(r'[.،؛!؟]', text)
        
        # Filter empty sentences and trim
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If we have very few sentences, try to split by common conjunctions
        if len(sentences) <= 2 and any(len(s) > 100 for s in sentences):
            expanded_sentences = []
            for sentence in sentences:
                if len(sentence) > 100:
                    # Split by conjunctions but only if resulting segments are substantial
                    parts = re.split(r'\s+و\s+|\s+ثم\s+|\s+أو\s+', sentence)
                    substantial_parts = [p for p in parts if len(p) > 30]
                    if substantial_parts:
                        expanded_sentences.extend(substantial_parts)
                    else:
                        expanded_sentences.append(sentence)
                else:
                    expanded_sentences.append(sentence)
            sentences = expanded_sentences
        
        return sentences
    
    @staticmethod
    def detect_arabic_dialect(text: str) -> str:
        """
        Simple heuristic-based dialect detection for Arabic text.
        Returns dialect name or 'MSA' for Modern Standard Arabic.
        This is a simplified approach and could be improved with ML models.
        """
        # Normalize text
        text = ArabicTextProcessor.normalize_arabic(text.lower())
        
        # Check for Egyptian dialect markers
        egyptian_markers = ['احنا', 'انتو', 'ازيك', 'عايز', 'دلوقتي', 'بتاع', 'ازاي', 'كده', 'فين']
        if any(marker in text for marker in egyptian_markers):
            return 'Egyptian'
            
        # Check for Levantine dialect markers
        levantine_markers = ['هيك', 'هلق', 'شو', 'كيفك', 'منيح', 'بدي', 'حكي']
        if any(marker in text for marker in levantine_markers):
            return 'Levantine'
            
        # Check for Gulf dialect markers
        gulf_markers = ['شلونك', 'الحين', 'وش', 'يبغى', 'تبين', 'چذي', 'شفيك', 'وايد']
        if any(marker in text for marker in gulf_markers):
            return 'Gulf'
            
        # Check for Maghrebi dialect markers
        maghrebi_markers = ['واش', 'كيفاش', 'بزاف', 'ماشي', 'لابس', 'غادي']
        if any(marker in text for marker in maghrebi_markers):
            return 'Maghrebi'
            
        # Default to MSA (Modern Standard Arabic)
        return 'MSA'
    
    @staticmethod
    def generate_query_variants(query: str, max_variants: int = 3) -> List[str]:
        """
        Generate multiple variants of an Arabic query to improve recall.
        Handles morphological variations and common dialectical differences.
        """
        if not query or not ArabicTextProcessor.contains_arabic(query):
            return [query]
            
        # Start with basic preprocessing
        query_data = ArabicTextProcessor.preprocess_arabic_query(query)
        variants = [query_data["original"], query_data["normalized"]]
        
        # Add stemmed version
        if query_data["stemmed"] != query_data["normalized"]:
            variants.append(query_data["stemmed"])
            
        # Create variants with common dialectical transformations
        dialect = ArabicTextProcessor.detect_arabic_dialect(query)
        normalized = query_data["normalized"]
        
        if dialect == 'Egyptian':
            # Convert ث to س (common in Egyptian)
            variants.append(normalized.replace('ث', 'س'))
            # Convert ذ to ز
            variants.append(normalized.replace('ذ', 'ز'))
        elif dialect == 'Gulf':
            # Convert ج to ي in some Gulf dialects
            variants.append(normalized.replace('ج', 'ي'))
            # Convert ك to تش in some contexts
            if 'ك' in normalized:
                variants.append(normalized.replace('ك', 'تش'))
                
        # Filter duplicates and limit
        unique_variants = []
        seen = set()
        for v in variants:
            if v not in seen and v.strip():
                unique_variants.append(v)
                seen.add(v)
                if len(unique_variants) >= max_variants:
                    break
                    
        return unique_variants
    
    @staticmethod
    def are_arabic_words_related(word1: str, word2: str) -> bool:
        """
        Check if two Arabic words are likely morphologically related.
        Uses a simplified root-based approach by comparing word stems
        and checking for shared character patterns.
        """
        # Normalize both words
        w1 = ArabicTextProcessor.normalize_arabic(word1)
        w2 = ArabicTextProcessor.normalize_arabic(word2)
        
        # Exact match
        if w1 == w2:
            return True
            
        # Apply stemming
        s1 = ArabicTextProcessor.light_stem_arabic(w1)
        s2 = ArabicTextProcessor.light_stem_arabic(w2)
        
        # Check stemmed versions
        if s1 == s2:
            return True
            
        # Check if one is contained in the other
        if (s1 in s2 and len(s1) >= 3) or (s2 in s1 and len(s2) >= 3):
            return True
            
        # For short words (3-4 chars), check if they share most characters in same order
        if 3 <= len(s1) <= 4 and 3 <= len(s2) <= 4:
            common_chars = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
            if common_chars >= min(len(s1), len(s2)) - 1:
                return True
                
        return False
