import re
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm.factory import ModelFactory
from core.language.arabic_utils import ArabicTextProcessor
import numpy as np
from loguru import logger
import asyncio
from functools import lru_cache

class ContextualChunker:
    def __init__(self, 
                chunk_size: int = 800,  # Smaller chunks for better Arabic processing
                chunk_overlap: int = 250,  # Increased overlap for better context
                arabic_optimized: bool = True,
                batch_size: int = 5,  # Process chunks in batches for better performance
                max_concurrency: int = 3):  # Maximum number of concurrent LLM calls
        self.llm = ModelFactory.create_llm()
        self.arabic_optimized = arabic_optimized
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        
        # Configure splitter with parameters better suited for Arabic text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "؟", "،", " ", ""]  # Arabic-friendly separators
        )
        
    async def process_with_context(self, content: str) -> List[Dict[str, Any]]:
        """Process document content with contextual understanding, optimized for Arabic"""
        # Detect if content is primarily Arabic
        is_arabic_content = ArabicTextProcessor.contains_arabic(content)
        
        # Performance optimization: Skip heavy processing for very large content
        if len(content) > 100000:  # For very large documents
            logger.info("Large document detected, using optimized chunking strategy")
            chunk_size = 1200  # Use larger chunks for efficiency
            chunk_overlap = 150  # Reduce overlap
            self.text_splitter.chunk_size = chunk_size
            self.text_splitter.chunk_overlap = chunk_overlap
        
        # Split into initial chunks - use appropriate method based on content language
        if is_arabic_content and self.arabic_optimized:
            base_chunks = self._split_arabic_content(content)
        else:
            base_chunks = self.text_splitter.split_text(content)
        
        logger.info(f"Created {len(base_chunks)} initial chunks from content")
        
        # Process chunks in batches to avoid overwhelming the LLM service
        contextualized_chunks = []
        for i in range(0, len(base_chunks), self.batch_size):
            batch = base_chunks[i:i+self.batch_size]
            tasks = []
            
            # Create tasks for each chunk in the current batch
            for j, chunk in enumerate(batch):
                chunk_idx = i + j
                prev_chunk = base_chunks[chunk_idx-1] if chunk_idx > 0 else ""
                next_chunk = base_chunks[chunk_idx+1] if chunk_idx < len(base_chunks)-1 else ""
                tasks.append(self._process_single_chunk(chunk, prev_chunk, next_chunk, chunk_idx))
            
            # Process batch concurrently with controlled concurrency
            batch_results = await asyncio.gather(*tasks)
            contextualized_chunks.extend(batch_results)
            
            # Log progress for long-running processes
            if i + self.batch_size < len(base_chunks):
                logger.debug(f"Processed {i + self.batch_size}/{len(base_chunks)} chunks")
        
        return contextualized_chunks
    
    async def _process_single_chunk(self, chunk: str, prev_chunk: str, next_chunk: str, chunk_idx: int) -> Dict[str, Any]:
        """Process a single chunk with context, with rate limiting"""
        # Detect if chunk contains Arabic
        is_arabic = ArabicTextProcessor.contains_arabic(chunk)
        
        try:
            # Use semaphore to limit concurrent LLM calls
            async with self.semaphore:
                # Generate contextual summary using LLM with language-specific prompting
                context = await self._generate_chunk_context(
                    chunk, prev_chunk, next_chunk, is_arabic=is_arabic
                )
                
                # Extract relevant entities and keywords for Arabic content
                if is_arabic:
                    keywords = ArabicTextProcessor.extract_arabic_keywords(chunk)
                    dialect = ArabicTextProcessor.detect_arabic_dialect(chunk)
                else:
                    keywords = []
                    dialect = None
                
                # Create enhanced chunk with richer metadata and vector-friendly attributes
                return {
                    "content": chunk,
                    "context": context,
                    "embedding_text": self._create_embedding_text(chunk, context, is_arabic),
                    "metadata": {
                        "chunk_index": chunk_idx,
                        "is_arabic": is_arabic,
                        "dialect": dialect if is_arabic else None,
                        "keywords": keywords[:5] if keywords else [],
                        "context_summary": context["summary"],
                        "language": "arabic" if is_arabic else "other",
                        "topics": context.get("topics", []),
                        "has_context": True  # Flag for retrieval filters
                    }
                }
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx}: {e}")
            # Add a basic version of the chunk to avoid data loss
            return {
                "content": chunk,
                "context": {"summary": "", "topics": [], "entities": []},
                "embedding_text": chunk,  # Fallback to original content
                "metadata": {"chunk_index": chunk_idx, "is_arabic": is_arabic, "error": str(e), "has_context": False}
            }
    
    def _create_embedding_text(self, chunk: str, context: Dict[str, Any], is_arabic: bool) -> str:
        """Create enhanced text for better embeddings by combining chunk content with context"""
        summary = context.get("summary", "").strip()
        topics = context.get("topics", [])
        
        if not summary:
            return chunk
            
        # For Arabic, use Arabic template
        if is_arabic:
            topic_text = " ".join(topics[:5]) if topics else ""
            return f"{summary}\n\nالمحتوى الأساسي:\n{chunk}\n\nالمواضيع الرئيسية: {topic_text}"
        
        # For other languages
        topic_text = ", ".join(topics[:5]) if topics else ""
        return f"{summary}\n\nMain content:\n{chunk}\n\nTopics: {topic_text}"
    
    def _split_arabic_content(self, content: str) -> List[str]:
        """Split Arabic content using specialized techniques for better semantic chunking"""
        # Use ArabicTextProcessor to segment the text if available
        try:
            segments = ArabicTextProcessor.segment_arabic_text(content)
            if segments and len(segments) > 1:
                return segments
        except Exception as e:
            logger.warning(f"Arabic segmentation error: {e}")
        
        # Fallback to standard splitter
        return self.text_splitter.split_text(content)
        
    async def _generate_chunk_context(
        self, 
        chunk: str, 
        prev_chunk: str, 
        next_chunk: str,
        is_arabic: bool = False
    ) -> Dict[str, Any]:
        """Generate contextual understanding of chunk with language awareness"""
        # Optimization: Use a simpler prompt for very short chunks
        if len(chunk) < 100:
            logger.debug("Using simplified context generation for very short chunk")
            return {
                "summary": chunk[:80] + "..." if len(chunk) > 80 else chunk,
                "topics": [],
                "entities": []
            }
            
        prompt = self._get_context_prompt(chunk, prev_chunk, next_chunk, is_arabic=is_arabic)
        response = await self.llm.agenerate([prompt])
        
        # Parse context from response
        context = self._parse_context_response(
            response.generations[0][0].text,
            is_arabic=is_arabic
        )
        return context

    def _get_context_prompt(self, 
                          chunk: str, 
                          prev_chunk: Optional[str] = None, 
                          next_chunk: Optional[str] = None,
                          is_arabic: bool = False) -> str:
        """
        Generate a prompt for obtaining context information about the current chunk.
        Optimized for Arabic if the content contains Arabic text.
        
        Args:
            chunk: The current text chunk to generate context for
            prev_chunk: The preceding chunk text (if available)
            next_chunk: The following chunk text (if available)
            is_arabic: Whether the chunk contains Arabic text
            
        Returns:
            A formatted prompt string for context generation
        """
        if is_arabic:
            # Arabic-specific prompt
            prompt = """قم بإنشاء ملخص موجز للمحتوى التالي، مع تسليط الضوء على المواضيع الرئيسية والأفكار الأساسية. استخدم لغة عربية فصيحة وطبيعية:

"""
        else:
            prompt = "Please generate a brief summary of the following content, highlighting key topics and main ideas:\n\n"
        
        # Add context from surrounding chunks if available
        if prev_chunk:
            if is_arabic:
                prompt += "المحتوى السابق:\n" + prev_chunk + "\n\n"
            else:
                prompt += "Previous content:\n" + prev_chunk + "\n\n"
        
        if is_arabic:
            prompt += "المحتوى الحالي:\n" + chunk + "\n\n"
        else:
            prompt += "Current content:\n" + chunk + "\n\n"
        
        if next_chunk:
            if is_arabic:
                prompt += "المحتوى اللاحق:\n" + next_chunk + "\n\n"
            else:
                prompt += "Following content:\n" + next_chunk + "\n\n"
        
        if is_arabic:
            prompt += "\nقدم ملخصاً سياقياً موجزاً (2-3 جمل) يلتقط المعلومات الأساسية في هذا المحتوى. يجب أن يكون الملخص دقيقاً وبلغة عربية طبيعية وسلسة."
            prompt += "\nحدد أيضاً الكلمات المفتاحية الرئيسية (3-5 كلمات) والمفاهيم المهمة التي وردت في النص."
        else:
            prompt += "\nProvide a concise context summary (2-3 sentences) that captures the essential information in this content."
        
        return prompt

    def _parse_context_response(self, response_text: str, is_arabic: bool = False) -> Dict[str, Any]:
        """
        Parse the LLM response to extract contextual information.
        Enhanced for Arabic content processing.
        
        Args:
            response_text: The response text from the LLM
            is_arabic: Whether the response is in Arabic
            
        Returns:
            Dictionary containing parsed context information
        """
        # Clean and extract the response
        summary = response_text.strip()
        
        # Create a structured context object
        context = {
            "summary": summary,
            "topics": [],
            "entities": []
        }
        
        # Extract key topics - with Arabic-specific handling if needed
        try:
            if is_arabic:
                # For Arabic, use more specialized splitting with Arabic punctuation
                sentences = re.split(r'[.؟!،;]', summary)
                # Try to extract keywords from either explicit keyword section or from the summary
                keywords_section = None
                
                # Look for keywords section (might be after the summary)
                for sent in sentences:
                    if 'مفتاحية' in sent or 'رئيسية' in sent:
                        keywords_section = sent
                        break
                
                if keywords_section:
                    # Extract keywords from the dedicated section
                    keyword_candidates = [k.strip() for k in re.split(r'[,،:;]', keywords_section) 
                                         if k.strip() and not any(x in k for x in ['مفتاحية', 'رئيسية', 'كلمات'])]
                    context["topics"] = [k for k in keyword_candidates if len(k) > 1][:5]
                else:
                    # Extract from first sentence as fallback
                    if sentences and len(sentences) > 0:
                        first_sentence = sentences[0].strip()
                        important_terms = [term.strip() for term in re.split(r'[,،]', first_sentence) if term.strip()]
                        context["topics"] = important_terms[:3]  # Take just a few terms
                        
                # Try to extract named entities if any markers exist
                entities = []
                entity_markers = ['اسم', 'شخص', 'مكان', 'موقع', 'شركة', 'منظمة', 'مؤسسة']
                for marker in entity_markers:
                    if marker in summary:
                        # Find potential entities near this marker
                        pattern = rf'{marker}\s+([^\.,،؟!]+)'
                        matches = re.findall(pattern, summary)
                        entities.extend([m.strip() for m in matches if m.strip()])
                
                context["entities"] = entities
            else:
                # Standard English processing
                sentences = summary.split('.')
                if len(sentences) > 1:
                    first_sentence = sentences[0].strip()
                    topics = [phrase.strip() for phrase in first_sentence.split(',') if phrase.strip()]
                    if topics:
                        context["topics"] = topics
        except Exception as e:
            logger.error(f"Error parsing context: {e}")
            # If extraction fails, leave as empty
            pass
            
        return context