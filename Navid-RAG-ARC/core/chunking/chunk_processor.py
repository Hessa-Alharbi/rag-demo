from typing import List, Dict, Any, Optional
from core.chunking.contextual_chunker import ContextualChunker
from core.language.arabic_utils import ArabicTextProcessor
from loguru import logger
import asyncio
import time

class ChunkProcessor:
    """
    Main processing pipeline for document chunking with optimized flow
    """
    def __init__(self):
        self.contextual_chunker = ContextualChunker(
            chunk_size=800,  # Optimized chunk size for retrieval 
            chunk_overlap=250,
            arabic_optimized=True,
            batch_size=5,
            max_concurrency=3  # Control concurrent processing
        )
        
    async def process_document(self, content: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a document through the complete chunking pipeline
        
        Args:
            content: The document content to process
            doc_id: Document identifier
            metadata: Additional document metadata
            
        Returns:
            List of processed chunks ready for indexing
        """
        start_time = time.time()
        
        if not content or not content.strip():
            logger.warning(f"Empty content for document {doc_id}")
            return []
            
        # Detect document language
        is_arabic = ArabicTextProcessor.contains_arabic(content)
        language = "arabic" if is_arabic else "other"
        
        logger.info(f"Processing document {doc_id} ({len(content)} chars, language: {language})")
        
        # Step 1: Create contextualized chunks
        contextualized_chunks = await self.contextual_chunker.process_with_context(content)
        
        # Step 2: Enhance chunks with document-level metadata
        enhanced_chunks = []
        doc_metadata = metadata or {}
        
        for i, chunk in enumerate(contextualized_chunks):
            # Add index position
            chunk_position = i / max(1, len(contextualized_chunks) - 1)  # Normalized position (0-1)
            
            # Create final metadata
            final_metadata = {
                **chunk.get("metadata", {}),
                "doc_id": doc_id,
                "doc_metadata": doc_metadata,
                "chunk_position": chunk_position,  # Useful for chronological ordering
                "total_chunks": len(contextualized_chunks),
                "language": language
            }
            
            # Create optimized embedding text that combines content + context
            embedding_text = chunk.get("embedding_text", self._create_optimized_embedding_text(chunk))
            
            enhanced_chunks.append({
                "content": chunk["content"],
                "context": chunk["context"],
                "embedding_text": embedding_text,
                "metadata": final_metadata
            })
            
        processing_time = time.time() - start_time
        logger.info(f"Document {doc_id} processed in {processing_time:.2f}s: {len(enhanced_chunks)} chunks created")
        
        return enhanced_chunks
        
    def _create_optimized_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """Create optimized text for vector embedding that captures content and context"""
        content = chunk.get("content", "")
        context = chunk.get("context", {})
        summary = context.get("summary", "") if context else ""
        
        # If no summary, return original content
        if not summary:
            return content
            
        # Check if content is Arabic
        is_arabic = chunk.get("metadata", {}).get("is_arabic", False)
        
        if is_arabic:
            # For Arabic content, combine content with Arabic-formatted context
            keywords = chunk.get("metadata", {}).get("keywords", [])
            keywords_text = " ".join(keywords) if keywords else ""
            
            # Create Arabic optimized embedding text
            return f"{summary}\n\n{content}\n\nالكلمات المفتاحية: {keywords_text}"
        else:
            # For other languages
            topics = context.get("topics", [])
            topics_text = ", ".join(topics) if topics else ""
            
            # Create standard optimized embedding text
            return f"{summary}\n\n{content}\n\nTopics: {topics_text}"
