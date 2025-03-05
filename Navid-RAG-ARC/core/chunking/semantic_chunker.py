"""
Advanced semantic chunking strategies for optimal text splitting and retrieval.
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
import re
import nltk
from nltk.tokenize import sent_tokenize
from core.language.arabic_utils import ArabicTextProcessor

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)

class SemanticChunker:
    """
    Advanced text chunking with semantic boundary detection and multi-language support.
    Optimizes chunks for better retrieval in RAG applications.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_headings: bool = True,
        respect_semantic_units: bool = True
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            preserve_headings: Whether to keep headings with their content
            respect_semantic_units: Whether to try to keep semantic units intact
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_headings = preserve_headings
        self.respect_semantic_units = respect_semantic_units
        
        # Initialize base text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                # Try to split on semantic boundaries first
                "\n\n\n",  # Multiple line breaks often indicate section boundaries
                "\n\n",    # Double line breaks often indicate paragraph boundaries
                "\n",      # Single line breaks
                ". ",      # End of sentences
                "! ",      # End of sentences
                "? ",      # End of sentences
                ";",       # Semi-colon
                ",",       # Comma
                " ",       # Space
                ""         # Any character
            ]
        )
        
    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: The input text to split
            
        Returns:
            List of text chunks
        """
        # Clean and preprocess the text
        text = self._clean_text(text)
        
        # Check for Arabic content and apply special handling
        if ArabicTextProcessor.contains_arabic(text):
            return self._split_arabic_text(text)
        
        # For regular text, apply semantic boundary detection
        if self.respect_semantic_units:
            return self._split_with_semantic_boundaries(text)
        
        # Fallback to standard chunking
        return self.text_splitter.split_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text before chunking"""
        if not text:
            return ""
            
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix broken sentences (common in PDFs)
        text = re.sub(r'(\w)- (\w)', r'\1\2', text)
        
        # Fix sentence boundaries without proper spacing
        text = re.sub(r'(\.)(\w)', r'\1 \2', text)
        
        return text.strip()
    
    def _split_with_semantic_boundaries(self, text: str) -> List[str]:
        """Split text respecting semantic boundaries like sentences and paragraphs"""
        # First detect section headings
        if self.preserve_headings:
            sections = self._split_by_headings(text)
            if len(sections) > 1:
                result = []
                for section in sections:
                    # Process each section separately but keep headings with their content
                    if len(section) <= self.chunk_size:
                        result.append(section)
                    else:
                        # Split large sections further but try to preserve paragraph structure
                        section_chunks = self._split_by_paragraphs(section)
                        result.extend(section_chunks)
                return result
        
        # If no clear section structure, try paragraph-based chunking
        paragraphs = self._split_by_paragraphs(text)
        if len(paragraphs) > 1:
            return paragraphs
        
        # Fall back to sentence-based chunking for long text without paragraph structure
        if len(text) > self.chunk_size:
            return self._split_by_sentences(text)
            
        # If text is already small enough, return as is
        if len(text) <= self.chunk_size:
            return [text]
            
        # Final fallback to standard chunker
        return self.text_splitter.split_text(text)
    
    def _split_by_headings(self, text: str) -> List[str]:
        """Split text by section headings"""
        # Match common heading patterns (e.g., "## Heading", "1.1 Heading", "Section 1:", etc.)
        heading_pattern = re.compile(r'(?:\n|\A)(?:#+\s+|(?:\d+\.)+\s+|Section\s+\d+:?\s+|Chapter\s+\d+:?\s+)([^\n]+)')
        
        matches = list(heading_pattern.finditer(text))
        if not matches:
            return [text]
            
        sections = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i < len(matches)-1 else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append(section_text)
                
        # If no valid sections found, return original text
        return sections if sections else [text]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs, combining short paragraphs if needed"""
        # Split by double newlines which typically indicate paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        result = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size, start a new chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                result.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    
        # Add the last chunk if it exists
        if current_chunk:
            result.append(current_chunk.strip())
            
        # If we ended up with no chunks or one large chunk, fall back to the text splitter
        if not result or (len(result) == 1 and len(result[0]) > self.chunk_size):
            return self.text_splitter.split_text(text)
            
        return result
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences, combining short sentences if needed"""
        try:
            sentences = sent_tokenize(text)
            
            result = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence would exceed chunk size, start a new chunk
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    result.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                        
            # Add the last chunk if it exists
            if current_chunk:
                result.append(current_chunk.strip())
                
            return result
            
        except Exception as e:
            logger.warning(f"Error in sentence tokenization: {e}. Falling back to basic splitter.")
            return self.text_splitter.split_text(text)
    
    def _split_arabic_text(self, text: str) -> List[str]:
        """Special handling for Arabic text"""
        # Normalize Arabic text first
        normalized_text = ArabicTextProcessor.normalize_arabic(text)
        
        # Use specialized Arabic paragraph splitting (Arabic uses different paragraph markers)
        paragraphs = re.split(r'(?:\n\s*\n|\r\n\s*\r\n)', normalized_text)
        
        result = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed chunk size, start a new chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                result.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    
        # Add the last chunk if it exists
        if current_chunk:
            result.append(current_chunk.strip())
            
        # If we ended up with no chunks or one large chunk, fall back to the text splitter
        if not result or (len(result) == 1 and len(result[0]) > self.chunk_size):
            return self.text_splitter.split_text(text)
            
        return result

    def generate_chunk_titles(self, chunks: List[str]) -> List[str]:
        """Generate descriptive titles for chunks"""
        titles = []
        
        for chunk in chunks:
            # Try to extract meaningful title from first line
            lines = chunk.strip().split('\n')
            first_line = lines[0] if lines else ""
            
            # If first line looks like a heading, use it
            if first_line and len(first_line) <= 100 and not first_line.endswith(('.', '?', '!')):
                title = first_line
            else:
                # Extract first sentence or first 50 chars
                match = re.search(r'^([^.!?]+[.!?])', chunk)
                if match:
                    title = match.group(1)
                    if len(title) > 60:
                        title = title[:57] + "..."
                else:
                    title = chunk[:50] + "..." if len(chunk) > 50 else chunk
                    
            titles.append(title)
            
        return titles
