import re
from typing import Dict, List, Tuple, Any
from loguru import logger

class ResponseAnalyzer:
    """
    Analyzes the quality of LLM-generated responses and provides metrics
    for response improvement.
    """
    
    @staticmethod
    def analyze_quality(response: str, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of a response based on various metrics.
        
        Args:
            response: The generated response to analyze
            query: The original user query
            context_docs: The context documents used for generation
            
        Returns:
            Dictionary of quality metrics
        """
        if not response:
            return {"quality_score": 0, "issues": ["Empty response"]}
            
        metrics = {}
        issues = []
        
        # 1. Analyze verbosity
        verbosity_score, verbosity_issues = ResponseAnalyzer._analyze_verbosity(response)
        metrics["verbosity_score"] = verbosity_score
        issues.extend(verbosity_issues)
        
        # 2. Analyze repetition
        repetition_score, repetition_issues = ResponseAnalyzer._analyze_repetition(response)
        metrics["repetition_score"] = repetition_score
        issues.extend(repetition_issues)
        
        # 3. Analyze coherence
        coherence_score, coherence_issues = ResponseAnalyzer._analyze_coherence(response)
        metrics["coherence_score"] = coherence_score
        issues.extend(coherence_issues)
        
        # 4. Analyze relevance to query
        relevance_score, relevance_issues = ResponseAnalyzer._analyze_relevance(response, query)
        metrics["relevance_score"] = relevance_score
        issues.extend(relevance_issues)
        
        # 5. Analyze information accuracy
        accuracy_score, accuracy_issues = ResponseAnalyzer._analyze_accuracy(response, context_docs)
        metrics["accuracy_score"] = accuracy_score
        issues.extend(accuracy_issues)
        
        # Calculate overall quality score
        quality_score = (
            verbosity_score * 0.2 + 
            repetition_score * 0.3 + 
            coherence_score * 0.2 + 
            relevance_score * 0.2 + 
            accuracy_score * 0.1
        )
        
        metrics["quality_score"] = round(quality_score, 2)
        metrics["issues"] = issues
        
        return metrics
    
    @staticmethod
    def _analyze_verbosity(response: str) -> Tuple[float, List[str]]:
        """Analyze the verbosity of the response"""
        issues = []
        
        # Check response length
        words = response.split()
        num_words = len(words)
        
        if num_words < 10:
            issues.append("Response too short")
            return 0.5, issues
            
        if num_words > 200:
            issues.append("Response excessively verbose")
            return 0.3, issues
            
        # Check for unnecessary phrases
        filler_phrases = [
            "I would like to inform you that",
            "It is important to note that",
            "I would like to point out that",
            "Based on the context provided",
            "According to the information provided",
            "As stated in the context",
            "As mentioned earlier",
            "من الجدير بالذكر أن",
            "كما ورد في المعلومات المقدمة",
            "وفقا للمعلومات المتاحة",
        ]
        
        filler_count = 0
        for phrase in filler_phrases:
            if phrase.lower() in response.lower():
                filler_count += 1
                issues.append(f"Contains filler phrase: '{phrase}'")
                
        # Calculate verbosity score (1.0 is best)
        if filler_count > 0:
            verbosity_score = max(0.0, 1.0 - (filler_count * 0.2))
        else:
            # Good length without fillers
            verbosity_score = 1.0 if 20 <= num_words <= 150 else 0.7
            
        return verbosity_score, issues
    
    @staticmethod
    def _analyze_repetition(response: str) -> Tuple[float, List[str]]:
        """Analyze repetition in the response"""
        issues = []
        
        # Check for repeated words
        words = response.split()
        if len(words) <= 1:
            return 1.0, issues  # Very short responses get full score
            
        # Calculate word frequency
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Check for high-frequency words (excluding common stopwords)
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'to', 'in', 'is', 'it', 
                   'that', 'this', 'for', 'with', 'as', 'by', 'on', 'at', 'من', 'في', 'على', 
                   'إلى', 'عن', 'أن', 'التي', 'الذي', 'هذا', 'هذه', 'تلك', 'ذلك', 'مع', 'عند', 'و'}
        
        repeated_words = []
        for word, count in word_freq.items():
            if word not in stopwords and len(word) > 3 and count > 3:
                repeated_words.append((word, count))
                
        if repeated_words:
            issues.append(f"Frequent word repetition: {', '.join([f'{w}({c})' for w, c in repeated_words[:3]])}")
        
        # Check for repeated phrases (3+ words)
        phrase_repetition = ResponseAnalyzer._detect_phrase_repetition(response)
        if phrase_repetition:
            issues.append(f"Contains repeated phrases: {phrase_repetition[:2]}")
            
        # Check for duplicate sentences
        sentences = re.split(r'[.!?،؛]', response)
        unique_sentences = set()
        duplicate_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:
                if sentence in unique_sentences:
                    duplicate_sentences.append(sentence[:30] + "...")
                else:
                    unique_sentences.add(sentence)
                    
        if duplicate_sentences:
            issues.append(f"Contains {len(duplicate_sentences)} duplicate sentences")
            
        # Calculate repetition score (1.0 is best)
        repetition_factors = len(repeated_words) + len(phrase_repetition) + len(duplicate_sentences)
        repetition_score = max(0.0, 1.0 - (repetition_factors * 0.2))
        
        return repetition_score, issues
    
    @staticmethod
    def _detect_phrase_repetition(text: str) -> List[str]:
        """Detect repeated phrases in text"""
        words = text.split()
        repeated_phrases = []
        
        # Look for phrases of length 3-6 words
        for phrase_len in range(3, 7):
            if len(words) < phrase_len * 2:
                continue
                
            # Check every possible starting position
            for i in range(len(words) - phrase_len * 2 + 1):
                phrase1 = ' '.join(words[i:i+phrase_len])
                
                # Look for this phrase elsewhere in the text
                for j in range(i + phrase_len, len(words) - phrase_len + 1):
                    phrase2 = ' '.join(words[j:j+phrase_len])
                    
                    if phrase1 == phrase2 and phrase1 not in repeated_phrases:
                        repeated_phrases.append(phrase1)
                        
        return repeated_phrases
    
    @staticmethod
    def _analyze_coherence(response: str) -> Tuple[float, List[str]]:
        """Analyze the coherence of the response"""
        issues = []
        
        # Check for minimal response length required for coherence analysis
        if len(response.split()) < 15:
            return 0.8, issues  # Short responses are assumed to be coherent
            
        # Check for connective words indicating logical flow
        connectives = ['however', 'therefore', 'thus', 'because', 'since', 'although', 
                     'consequently', 'furthermore', 'moreover', 'لذلك', 'نظراً', 'حيث', 
                     'بالإضافة', 'ومع ذلك', 'أيضاً', 'بسبب', 'نتيجة']
                     
        has_connectives = any(conn in response.lower() for conn in connectives)
        
        # Check for abrupt topic shifts (heuristic approach)
        topics_shift = False
        sentences = re.split(r'[.!?،؛]', response)
        
        if len(sentences) >= 3:
            # Simple heuristic: check if consecutive sentences share common words
            common_words_threshold = 0.2  # At least 20% word overlap
            
            for i in range(len(sentences) - 1):
                s1 = sentences[i].strip().lower()
                s2 = sentences[i+1].strip().lower()
                
                if len(s1) < 5 or len(s2) < 5:
                    continue
                    
                words1 = set(s1.split())
                words2 = set(s2.split())
                
                # Calculate word overlap
                common_words = words1.intersection(words2)
                overlap_ratio = len(common_words) / min(len(words1), len(words2))
                
                if overlap_ratio < common_words_threshold:
                    topics_shift = True
                    issues.append("Detected abrupt topic shift between sentences")
                    break
        
        # Calculate coherence score
        if topics_shift:
            coherence_score = 0.4
        elif has_connectives:
            coherence_score = 1.0
        else:
            coherence_score = 0.7
            issues.append("No connective words for logical flow")
            
        return coherence_score, issues
    
    @staticmethod
    def _analyze_relevance(response: str, query: str) -> Tuple[float, List[str]]:
        """Analyze the relevance of the response to the query"""
        issues = []
        
        # Extract key terms from query
        query_terms = re.findall(r'\b\w+\b', query.lower())
        # Remove common stopwords
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'to', 'in', 'is', 'it', 
                   'that', 'this', 'for', 'with', 'as', 'by', 'on', 'at', 'من', 'في', 'على', 
                   'إلى', 'عن', 'أن', 'التي', 'الذي', 'هذا', 'هذه', 'تلك', 'ذلك', 'مع'}
        query_terms = [term for term in query_terms if term not in stopwords and len(term) > 2]
        
        if not query_terms:
            return 0.8, issues  # Can't determine relevance for queries without key terms
            
        # Check if key query terms appear in the response
        response_lower = response.lower()
        found_terms = [term for term in query_terms if term in response_lower]
        
        term_coverage = len(found_terms) / len(query_terms) if query_terms else 0
        
        if term_coverage < 0.5:
            issues.append(f"Response covers only {len(found_terms)}/{len(query_terms)} query terms")
            
        # Check for off-topic responses
        if len(found_terms) == 0 and len(query_terms) >= 2:
            issues.append("Response appears to be off-topic (no key query terms found)")
            relevance_score = 0.1
        else:
            # Calculate relevance score
            relevance_score = term_coverage
            
        return relevance_score, issues
    
    @staticmethod
    def _analyze_accuracy(response: str, context_docs: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """Analyze if the response matches factual information in the context"""
        issues = []
        
        # This is a simplistic check - full accuracy validation would require more advanced NLP
        # Here we just check if the response contains text segments from the context
        
        # Extract sample text from context documents
        context_samples = []
        for doc in context_docs:
            content = doc.get("content", "")
            if content:
                # Take the first 100 characters as a sample
                sample = content[:100].strip()
                context_samples.append(sample)
        
        if not context_samples:
            return 0.5, ["No context to validate accuracy"]
            
        # Check if response contains direct copies from context
        direct_copy = False
        for sample in context_samples:
            if len(sample) > 20 and sample in response:
                direct_copy = True
                issues.append("Response contains direct copy from context without reformulation")
                break
                
        # Basic accuracy score (proper accuracy checking requires deeper NLP)
        accuracy_score = 0.5 if direct_copy else 0.8
            
        return accuracy_score, issues
        
    @staticmethod
    def suggest_improvements(metrics: Dict[str, Any]) -> List[str]:
        """Suggest improvements based on quality analysis"""
        suggestions = []
        issues = metrics.get("issues", [])
        
        if metrics.get("verbosity_score", 1.0) < 0.7:
            suggestions.append("Make the response more concise by removing filler phrases and unnecessary context")
            
        if metrics.get("repetition_score", 1.0) < 0.7:
            suggestions.append("Eliminate repeated words, phrases, and duplicate sentences")
            
        if metrics.get("coherence_score", 1.0) < 0.7:
            suggestions.append("Improve logical flow with transition words and ensure sentences connect to each other")
            
        if metrics.get("relevance_score", 1.0) < 0.7:
            suggestions.append("Ensure the response directly addresses the key terms and intent of the query")
            
        # Add specific suggestions based on issues
        for issue in issues:
            if "filler phrase" in issue:
                suggestions.append("Remove introductory phrases like 'It is important to note that'")
            elif "Frequent word repetition" in issue:
                suggestions.append("Use synonyms to avoid word repetition")
            elif "abrupt topic shift" in issue:
                suggestions.append("Add transition sentences between different topics")
            elif "direct copy" in issue:
                suggestions.append("Reformulate information from sources instead of copying directly")
                
        return suggestions 