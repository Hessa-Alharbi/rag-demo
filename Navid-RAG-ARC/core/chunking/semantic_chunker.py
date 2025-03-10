"""
Advanced semantic chunking with specialized support for Arabic language documents.
Implements intelligent chunking that preserves semantic structure and context.
"""
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from loguru import logger
from core.language.arabic_utils import ArabicTextProcessor

class SemanticChunker:
    """
    Advanced document chunker that preserves semantic structure and boundaries.
    - Respects paragraph, section and heading boundaries
    - Handles Arabic text with specialized processing
    - Prevents cutting in the middle of sentences
    - Creates semantically meaningful chunks
    - Generates informative chunk titles
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        preserve_headings: bool = True,
        respect_semantic_units: bool = True,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000,
        language_sensitive: bool = True
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            preserve_headings: Whether to keep headings with their content
            respect_semantic_units: Whether to avoid breaking semantic units like paragraphs
            min_chunk_size: Minimum allowed chunk size in characters
            max_chunk_size: Maximum allowed chunk size in characters
            language_sensitive: Whether to apply language-specific chunking rules
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_headings = preserve_headings
        self.respect_semantic_units = respect_semantic_units
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.language_sensitive = language_sensitive
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: The document text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Detect if the text contains Arabic
        has_arabic = ArabicTextProcessor.contains_arabic(text)
        
        # Apply specific chunking strategy based on content
        if has_arabic and self.language_sensitive:
            return self._split_arabic_text(text)
        else:
            return self._split_general_text(text)
    
    def _split_general_text(self, text: str) -> List[str]:
        """
        Split general text into chunks respecting semantic boundaries.
        
        Args:
            text: Text to split
            
        Returns:
            List of chunks
        """
        # Find all semantic boundaries (headings, paragraphs, lists)
        sections = self._split_into_semantic_units(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Special handling for headings
        current_heading = None
        
        # Process each semantic section
        for section in sections:
            section_text = section.strip()
            if not section_text:
                continue
                
            section_size = len(section_text)
            
            # Check if this is a heading
            is_heading = self._is_heading(section_text)
            
            if is_heading and self.preserve_headings:
                # If we have a new heading, remember it
                current_heading = section_text
                
                # If heading is long enough, add it as its own chunk
                if section_size > self.min_chunk_size:
                    chunks.append(section_text)
                    current_heading = None
                continue
            
            # Check if adding this section would exceed max chunk size
            if current_size + section_size > self.max_chunk_size:
                # Finalize current chunk if it's not empty
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # If the section itself is too large, split it further
                if section_size > self.max_chunk_size:
                    # For large sections, split by sentences
                    sentences = self._split_into_sentences(section_text)
                    sentence_chunks = self._create_chunks_from_sentences(sentences)
                    chunks.extend(sentence_chunks)
                    continue
                
                # Include current heading with new chunk if needed
                if current_heading:
                    current_chunk.append(current_heading)
                    current_size = len(current_heading)
                    current_heading = None
            
            # Add the section to current chunk
            current_chunk.append(section_text)
            current_size += section_size
            
            # If we're approaching max size, finalize the chunk
            if current_size >= self.chunk_size:
                chunks.append("\n".join(current_chunk))
                
                # Prepare for next chunk with overlap
                if self.chunk_overlap > 0 and self.respect_semantic_units:
                    # For overlap, take the last few sections based on overlap size
                    overlap_size = 0
                    overlap_sections = []
                    
                    for section in reversed(current_chunk):
                        overlap_sections.insert(0, section)
                        overlap_size += len(section)
                        if overlap_size >= self.chunk_overlap:
                            break
                    
                    current_chunk = overlap_sections
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks
    
    def _split_arabic_text(self, text: str) -> List[str]:
        """
        Split Arabic text with specialized handling for Arabic language features.
        
        Args:
            text: Arabic text to split
            
        Returns:
            List of chunks
        """
        # First normalize the Arabic text
        try:
            normalized_text = ArabicTextProcessor.normalize_arabic(text)
        except Exception as e:
            logger.warning(f"Error normalizing Arabic text: {e}")
            normalized_text = text
        
        # Try to segment Arabic text using specialized segmentation
        # This respects Arabic sentence boundaries and conjunctions
        try:
            arabic_segments = ArabicTextProcessor.segment_arabic_text(normalized_text)
            
            # If segmentation returns meaningful segments, use those as the base units
            if arabic_segments and len(arabic_segments) > 1:
                return self._create_chunks_from_arabic_segments(arabic_segments)
        except Exception as e:
            logger.warning(f"Arabic segmentation failed: {e}")
        
        # Fallback: find Arabic-specific semantic boundaries
        # Look for section markers common in Arabic texts
        section_markers = [
            r'[.؟!]',                           # Sentence boundaries
            r'(\n\s*\n)',                       # Paragraph breaks
            r'(^|\n)[\u0621-\u064A\s]+?:',      # Arabic section headers with colon
            r'(\n\s*-+\s*\n)',                  # Separator lines
            r'(\n\s*\d+[\.-]\s*)'               # Numbered items
        ]
        
        # Use regex to split by these boundaries
        pattern = '|'.join(section_markers)
        sections = re.split(pattern, normalized_text)
        
        # Filter and clean sections
        sections = [s.strip() for s in sections if s and s.strip()]
        
        return self._create_chunks_from_arabic_sections(sections)
    
    def _split_into_semantic_units(self, text: str) -> List[str]:
        """Split text into semantic units like paragraphs and list items."""
        # Split on paragraph breaks, headings, list items, and other structural elements
        section_splits = [
            r'(\n\s*\n)',                        # Paragraph breaks
            r'(^|\n)#+\s+[^\n]+',                # Markdown headings
            r'(^|\n)\s*\d+\.\s+[^\n]+',          # Numbered list items
            r'(^|\n)\s*[-*]\s+[^\n]+',           # Bullet list items
            r'(^|\n)[A-Z][A-Z\s]+(?:[^a-z]|$)',  # ALL CAPS HEADINGS
            r'(^|\n).*?\n[=-]{2,}',              # Underlined headings
            r'(^|\n)>\s+[^\n]+'                  # Blockquotes
        ]
        
        pattern = '|'.join(section_splits)
        
        # First try splitting by defined patterns
        sections = re.split(pattern, text)
        
        # Clean up and filter the sections
        cleaned_sections = []
        for section in sections:
            if section and section.strip():
                cleaned_sections.append(section.strip())
        
        # If we couldn't find good splits, fall back to paragraphs
        if len(cleaned_sections) <= 1:
            cleaned_sections = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # If still no good splits, go sentence by sentence
            if len(cleaned_sections) <= 1:
                sentences = self._split_into_sentences(text)
                
                # Group sentences into reasonable sections
                section, section_length = [], 0
                for sentence in sentences:
                    section.append(sentence)
                    section_length += len(sentence)
                    
                    if section_length >= 500:  # reasonable section size
                        cleaned_sections.append(' '.join(section))
                        section, section_length = [], 0
                
                if section:  # don't forget the last section
                    cleaned_sections.append(' '.join(section))
        
        return cleaned_sections
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling various punctuation marks."""
        # Handle multiple sentence-ending punctuations
        sentence_endings = r'(?<=[.!?;؟،])\s+'
        
        # Special handling for Arabic if detected
        if ArabicTextProcessor.contains_arabic(text):
            # Add Arabic punctuation to the pattern
            sentence_endings = r'(?<=[.!?;؟،])\s+'
        
        # Split by sentences
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunks_from_sentences(self, sentences: List[str]) -> List[str]:
        """Create reasonably-sized chunks from sentences."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If this sentence alone exceeds max size, split it further
            if sentence_length > self.max_chunk_size:
                # If we have a current chunk, finalize it
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by phrases
                phrase_pattern = r'(?<=[,;:\(\)\[\]،])\s+'
                phrases = re.split(phrase_pattern, sentence)
                
                temp_chunk = []
                temp_length = 0
                
                for phrase in phrases:
                    phrase_length = len(phrase)
                    
                    if temp_length + phrase_length > self.max_chunk_size:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                            temp_chunk = []
                            temp_length = 0
                    
                    temp_chunk.append(phrase)
                    temp_length += phrase_length
                    
                    if temp_length >= self.chunk_size:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = []
                        temp_length = 0
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                
                continue
            
            # Check if adding this sentence would exceed target chunk size
            if current_length + sentence_length > self.chunk_size:
                # Finalize current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Start new chunk with this sentence
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _create_chunks_from_arabic_segments(self, segments: List[str]) -> List[str]:
        """Create chunks from Arabic text segments with appropriate handling."""
        chunks = []
        current_chunk_segments = []
        current_chunk_length = 0
        
        for segment in segments:
            segment_length = len(segment)
            
            # If a single segment is too large, we need to split it further
            if segment_length > self.max_chunk_size:
                # Finalize current chunk if we have content
                if current_chunk_segments:
                    chunks.append("\n".join(current_chunk_segments))
                    current_chunk_segments = []
                    current_chunk_length = 0
                
                # Use sentence splitting for oversize segments
                sentences = re.split(r'[.؟!،;]', segment)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                sentence_chunks = self._create_chunks_from_sentences(sentences)
                chunks.extend(sentence_chunks)
                continue
            
            # Check if adding this segment would exceed chunk size
            if current_chunk_length + segment_length > self.chunk_size:
                # Finalize current chunk
                if current_chunk_segments:
                    chunks.append("\n".join(current_chunk_segments))
                
                # Start new chunk with current segment
                current_chunk_segments = [segment]
                current_chunk_length = segment_length
            else:
                # Add segment to current chunk
                current_chunk_segments.append(segment)
                current_chunk_length += segment_length
                
                # If we're close to the target size, finalize the chunk
                if current_chunk_length >= self.chunk_size * 0.9:
                    chunks.append("\n".join(current_chunk_segments))
                    
                    # Handle overlap for next chunk
                    if self.chunk_overlap > 0:
                        overlap_size = 0
                        overlap_segments = []
                        
                        # Take segments from the end until we meet overlap size
                        for seg in reversed(current_chunk_segments):
                            if overlap_size + len(seg) <= self.chunk_overlap:
                                overlap_segments.insert(0, seg)
                                overlap_size += len(seg)
                            else:
                                break
                        
                        current_chunk_segments = overlap_segments
                        current_chunk_length = overlap_size
                    else:
                        current_chunk_segments = []
                        current_chunk_length = 0
        
        # Add final chunk if not empty
        if current_chunk_segments:
            chunks.append("\n".join(current_chunk_segments))
        
        return chunks
    
    def _create_chunks_from_arabic_sections(self, sections: List[str]) -> List[str]:
        """Create chunks from Arabic sections with appropriate handling."""
        chunks = []
        current_sections = []
        current_length = 0
        
        for section in sections:
            section_length = len(section)
            
            # If a single section is too large, split it further
            if section_length > self.max_chunk_size:
                # Finalize current chunk if needed
                if current_sections:
                    chunks.append("\n".join(current_sections))
                    current_sections = []
                    current_length = 0
                
                # Split the large section into smaller parts based on punctuation
                parts = re.split(r'[.؟!،;]', section)
                parts = [p.strip() for p in parts if p.strip()]
                
                # Create chunks from these parts
                temp_parts = []
                temp_length = 0
                
                for part in parts:
                    part_length = len(part)
                    
                    if temp_length + part_length > self.chunk_size:
                        if temp_parts:
                            chunks.append(" ".join(temp_parts))
                            temp_parts = []
                            temp_length = 0
                    
                    temp_parts.append(part)
                    temp_length += part_length
                
                if temp_parts:
                    chunks.append(" ".join(temp_parts))
                
                continue
            
            # Check if adding this section would exceed target size
            if current_length + section_length > self.chunk_size:
                # Finalize current chunk
                if current_sections:
                    chunks.append("\n".join(current_sections))
                
                # Start new chunk with this section
                current_sections = [section]
                current_length = section_length
            else:
                # Add section to current chunk
                current_sections.append(section)
                current_length += section_length
        
        # Add final chunk if not empty
        if current_sections:
            chunks.append("\n".join(current_sections))
        
        return chunks
    
    def _is_heading(self, text: str) -> bool:
        """Detect if a text segment is a heading."""
        # Check for common heading patterns
        patterns = [
            r'^#+ ',                          # Markdown heading
            r'^[A-Z][A-Z\s]+$',               # ALL CAPS
            r'\n[=\-]{3,}$',                  # Underlined heading
            r'^[0-9]+\.\s+[A-Z]',             # Numbered headings
            r'^[A-Z][a-z]+\s*:\s*$',          # Title with colon
            r'^[^\n.]{1,60}$'                 # Short single line (potential heading)
        ]
        
        # Arabic heading patterns
        if ArabicTextProcessor.contains_arabic(text):
            patterns.extend([
                r'^[\u0621-\u064A\s]+:',      # Arabic text with colon
                r'^[\u0621-\u064A\s]{1,50}$'  # Short Arabic line (potential heading)
            ])
        
        # Check against patterns
        for pattern in patterns:
            if re.search(pattern, text):
                return True
                
        # Check for short text that doesn't end with punctuation
        if len(text) < 100 and not re.search(r'[.;:!?،؟]$', text):
            return True
            
        return False
    
    def generate_chunk_titles(self, chunks: List[str]) -> List[str]:
        """
        Generate meaningful titles for chunks to aid navigation and context.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of titles corresponding to the chunks
        """
        titles = []
        
        for i, chunk in enumerate(chunks):
            # Default title
            title = f"Chunk {i+1}"
            
            # Try to extract a meaningful title
            if chunk:
                # Check for headings first
                lines = chunk.split('\n')
                first_line = lines[0].strip()
                
                # If first line looks like a heading, use it
                if self._is_heading(first_line):
                    # Truncate if too long
                    if len(first_line) > 50:
                        title = first_line[:50] + "..."
                    else:
                        title = first_line
                else:
                    # Otherwise use first sentence or part of it
                    if ArabicTextProcessor.contains_arabic(chunk):
                        # For Arabic, get first sentence
                        sentences = re.split(r'[.؟!]', chunk)
                    else:
                        sentences = re.split(r'[.!?]', chunk)
                        
                    if sentences:
                        first_sentence = sentences[0].strip()
                        if first_sentence:
                            if len(first_sentence) > 40:
                                title = first_sentence[:40] + "..."
                            else:
                                title = first_sentence
            
            titles.append(title)
        
        return titles
    
    def extract_semantic_headers(self, text: str) -> List[str]:
        """
        Extract semantic headers/titles from text.
        Useful for building table of contents.
        
        Args:
            text: Document text
            
        Returns:
            List of headers found in the text
        """
        headers = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Check each line for heading patterns
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if self._is_heading(line):
                headers.append(line)
                
        return headers
