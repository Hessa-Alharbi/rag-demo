from typing import List, Dict, Any
from uuid import UUID
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from loguru import logger
from apps.chat.models import Attachment
from core.llm.factory import ModelFactory
from core.llm.prompt_templates import REACT_PROMPT_TEMPLATE, HYBRID_SEARCH_TEMPLATE, SEARCH_PROMPT_TEMPLATES
from core.config import get_settings
from .models import Document, DocumentProcessingEvent, DocumentProcessingStatus, DocumentStatus, Chunk
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
import docx2txt
import aiofiles
import json
from sqlmodel import Session, select
from langdetect import detect
import asyncio
import re
from core.vector_store.singleton import VectorStoreSingleton
from core.chunking.semantic_chunker import SemanticChunker
from core.search.reranker import QueryResultReranker
from core.search.query_processor import QueryProcessor
from core.language.arabic_utils import ArabicTextProcessor
from langchain_community.document_loaders import PyPDFLoader

class RAGService:
    def __init__(self):
        try:
            self.settings = get_settings()
            self._initialize_lock = asyncio.Lock()
            self._initialized = False
            
            # Initialize enhanced semantic chunker instead of basic text splitter
            self.semantic_chunker = SemanticChunker(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP,
                preserve_headings=True,
                respect_semantic_units=True
            )
            
            # Initialize advanced query processing and reranking components
            self.query_processor = QueryProcessor()
            self.reranker = QueryResultReranker()
            
            # Initialize LLM
            self.llm = ModelFactory.create_llm()
            self.chat_model = ModelFactory.create_llm()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RAGService: {str(e)}")

    async def initialize(self):
        """Initialize global vector store and reranker"""
        if not self._initialized:
            async with self._initialize_lock:
                if not self._initialized:
                    await VectorStoreSingleton.get_instance()
                    await self.reranker.initialize()
                    await self.query_processor.initialize()
                    self._initialized = True

    async def extract_file_content(self, file_path: str) -> str:
        """Extract text content from different file types"""
        ext = Path(file_path).suffix.lower()
        try:
            if (ext == '.pdf'):
                # Use PyPDFLoader instead of fitz
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                # Merge content of all pages with improved formatting
                text = "\n\n".join(
                    self.clean_and_format_text(page.page_content)
                    for page in pages
                )
                return text
            elif ext in ['.docx', '.doc']:
                return docx2txt.process(file_path)
            elif ext == '.txt':
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    return await f.read()
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error extracting content from file: {e}")
            raise

    async def process_document(self, doc: Document, file_path: str) -> Document:
        """Process document and add to vector store with improved semantic chunking"""
        try:
            # Get conversation-specific vector store
            vector_store = await VectorStoreSingleton.get_conversation_store(str(doc.conversation_id))
            
            # Extract and process content
            content = await self.extract_file_content(file_path)
            doc.content = content
            
            # Clean and semantically chunk content for better retrieval
            # Use the semantic chunker instead of basic text_splitter
            chunks = self.semantic_chunker.split_text(content)
            
            # Generate meaningful titles for chunks
            chunk_titles = self.semantic_chunker.generate_chunk_titles(chunks)
            
            # Prepare chunks for vector store with richer metadata
            texts = []
            metadatas = []
            db_chunks = []
            
            for i, (chunk_text, chunk_title) in enumerate(zip(chunks, chunk_titles)):
                # Get language of the chunk for better metadata
                chunk_lang = "unknown"
                try:
                    chunk_lang = detect(chunk_text[:100])  # Only use first 100 chars for language detection
                except:
                    pass
                
                # Create a chunk with enhanced metadata
                chunk = Chunk(
                    document_id=doc.id,
                    content=chunk_text,
                    chunk_metadata={
                        "index": i,
                        "source": file_path,
                        "title": chunk_title,
                        "doc_title": doc.title,
                        "language": chunk_lang,
                        "is_arabic": ArabicTextProcessor.contains_arabic(chunk_text)
                    }
                )
                db_chunks.append(chunk)
                texts.append(chunk_text)
                metadatas.append({
                    "document_id": str(doc.id),
                    "conversation_id": str(doc.conversation_id),
                    "chunk_index": i,
                    "source": file_path,
                    "title": chunk_title,
                    "doc_title": doc.title,
                    "language": chunk_lang,
                    "is_arabic": ArabicTextProcessor.contains_arabic(chunk_text)
                })
            
            # Add to vector store
            vector_ids = await vector_store.add_documents(texts, metadatas)
            
            # Update chunks and document
            for chunk, vector_id in zip(db_chunks, vector_ids):
                chunk.vector_id = vector_id
            
            doc.chunks.extend(db_chunks)
            doc.vector_ids = vector_ids
            doc.status = DocumentStatus.COMPLETED
            doc.is_searchable = True
            
            return doc
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            doc.status = DocumentStatus.FAILED
            raise

    def _preprocess_content(self, content: str) -> str:
        """Clean and preprocess document content for better chunking"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Fix common OCR issues
        content = re.sub(r'([a-z])(\.)([A-Z])', r'\1\2 \3', content)  # Fix sentence boundaries
        
        return content.strip()
        
    def _generate_chunk_title(self, chunk_text: str, index: int) -> str:
        """Generate a descriptive title for a chunk"""
        # Extract first line or first few words
        first_line = chunk_text.split('\n', 1)[0].strip()
        
        # Limit to reasonable length
        if len(first_line) > 50:
            first_line = first_line[:50] + "..."
            
        # If title is empty or just whitespace, use generic title
        if not first_line or first_line.isspace():
            return f"Chunk {index+1}"
            
        return first_line

    async def _process_document(self, doc: Document, file_path: str) -> Document:
        try:
            # Update status to processing
            await self._update_processing_status(
                doc, 
                DocumentProcessingStatus.PROCESSING,
                "Extracting document content"
            )
            
            # Extract content
            content = await self.extract_file_content(file_path)
            doc.content = content
            
            await self._update_processing_status(
                doc,
                DocumentProcessingStatus.VECTORIZING,
                "Creating document vectors"
            )
            
            # Process chunks and create vectors
            raw_docs = [
                LangchainDocument(
                    page_content=content,
                    metadata={"source": file_path}
                )
            ]

            # Split into chunks
            chunks = self.text_splitter.split_documents(raw_docs)
            
            # Create chunks in database
            db_chunks = []
            texts = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                db_chunk = Chunk(
                    document_id=doc.id,
                    content=chunk.page_content,
                    chunk_metadata={
                        "index": i,
                        **chunk.metadata
                    }
                )
                db_chunks.append(db_chunk)
                texts.append(chunk.page_content)
                metadatas.append({
                    "document_id": str(doc.id),
                    "conversation_id": str(doc.conversation_id),
                    "chunk_index": i
                })

            # Store vectors with explicit IDs
            vector_ids = await self.vector_store.add_documents(texts, metadatas)

            # Update chunks with vector IDs
            for chunk, vector_id in zip(db_chunks, vector_ids):
                chunk.vector_id = vector_id

            # Update document
            doc.vector_ids = vector_ids
            doc.status = DocumentStatus.COMPLETED
            doc.chunks.extend(db_chunks)

            # Update document status
            doc.current_status = DocumentProcessingStatus.COMPLETED
            doc.is_searchable = True
            await self._update_processing_status(
                doc,
                DocumentProcessingStatus.COMPLETED,
                "Document processing completed"
            )

            return doc

        except Exception as e:
            await self._update_processing_status(
                doc,
                DocumentProcessingStatus.FAILED,
                f"Processing failed: {str(e)}"
            )
            raise

    async def _update_processing_status(
        self,
        doc: Document,
        status: DocumentProcessingStatus,
        message: str,
        metadata: Dict = None
    ):
        """Update document processing status and create event"""
        doc.current_status = status
        event = DocumentProcessingEvent(
            document_id=doc.id,
            status=status,
            message=message,
            event_metadata=metadata or {}
        )
        doc.processing_events.append(event)

    async def can_query_document(self, doc_id: UUID, session: Session) -> bool:
        """Check if document is ready for querying"""
        doc = await self.get_document(doc_id, session)
        return doc and doc.is_searchable

    async def get_document_status(self, doc_id: UUID, session: Session) -> Dict:
        """Get document processing status and events"""
        doc = await self.get_document(doc_id, session)
        if not doc:
            raise ValueError("Document not found")
            
        return {
            "status": doc.current_status,
            "is_searchable": doc.is_searchable,
            "events": [
                {
                    "status": event.status,
                    "message": event.message,
                    "timestamp": event.created_at,
                    "metadata": event.event_metadata
                }
                for event in doc.processing_events
            ]
        }

    async def search_documents(
        self,
        query: str,
        filter_metadata: Dict[str, Any] = None,
        limit: int = 5
    ) -> List[dict]:
        try:
            # Pre-process query 
            processed_query = await self.query_processor.process_query(query)
            
            # Use primary processed query for search
            search_query = processed_query["processed_query"]
            
            results = await self.vector_store.similarity_search(
                search_query,
                k=limit,
                filter=filter_metadata
            )
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}", exc_info=True)
            raise

    async def hybrid_search(
        self, 
        query: str,
        conversation_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Enhanced hybrid search with multi-stage retrieval and reranking"""
        try:
            # Make sure the service is initialized
            await self.initialize()
            
            # Get conversation-specific vector store
            vector_store = await VectorStoreSingleton.get_conversation_store(conversation_id)
            
            if not query.strip():
                return []
            
            # Use advanced query processing
            processed_query = await self.query_processor.process_query(query)
            logger.info(f"Processed query: {processed_query['processed_query']} (original: {query})")
            
            # Use expanded queries for better recall
            expanded_queries = processed_query["expanded_queries"]
            if not expanded_queries:
                expanded_queries = [processed_query["processed_query"]]
                
            # Add the original query as a fallback
            if query not in expanded_queries:
                expanded_queries.append(query)
                
            # Initialize results list
            all_results = []
                
            # Check if query contains Arabic and apply special handling
            is_arabic = ArabicTextProcessor.contains_arabic(query)
            if is_arabic:
                # Use Arabic-specific query processing
                query_data = ArabicTextProcessor.preprocess_arabic_query(query)
                
                # Try normalized query first
                semantic_results = await vector_store.similarity_search(
                    query=query_data["normalized"],
                    k=limit * 2,  # Get more results for reranking
                    filter={"conversation_id": conversation_id}
                )
                
                all_results.extend(semantic_results)
                
                # Try Arabic keywords if normalized query returned few results
                if len(semantic_results) < limit and query_data["keywords"]:
                    keyword_query = " ".join(query_data["keywords"])
                    keyword_results = await vector_store.similarity_search(
                        query=keyword_query,
                        k=limit * 2,
                        filter={"conversation_id": conversation_id}
                    )
                    # Add results not already in the list
                    existing_ids = {r.get("id") for r in all_results if "id" in r}
                    all_results.extend([r for r in keyword_results if r.get("id") not in existing_ids])
            else:
                # For non-Arabic queries, use the expanded queries
                for search_query in expanded_queries[:3]:  # Limit to top 3 expanded queries
                    semantic_results = await vector_store.similarity_search(
                        query=search_query,
                        k=limit,  # Get more results for reranking
                        filter={"conversation_id": conversation_id}
                    )
                    
                    # Add unique results to the combined list
                    existing_ids = {r.get("id") for r in all_results if "id" in r}
                    all_results.extend([r for r in semantic_results if r.get("id") not in existing_ids])
                    
                    # If we have enough results, stop querying
                    if len(all_results) >= limit * 3:
                        break
            
            # If we still don't have enough results, try without conversation filter
            if len(all_results) < limit:
                # Try wider search with primary query
                wide_results = await vector_store.similarity_search(
                    query=processed_query["processed_query"],
                    k=limit,
                    filter=None  # Remove conversation filter for wider search
                )
                
                # Add unique results
                existing_ids = {r.get("id") for r in all_results if "id" in r}
                all_results.extend([r for r in wide_results if r.get("id") not in existing_ids])
            
            # If we have results, apply advanced ensemble reranking
            if all_results:
                # Use ensemble reranking for complex queries
                if processed_query["query_type"] == "complex" and len(query.split()) > 3:
                    reranked_results = await self.reranker.rerank_results(
                        query=query, 
                        results=all_results, 
                        top_k=limit,
                        method="ensemble"  # Use sophisticated ensemble method
                    )
                else:
                    # Use simpler keyword reranking for simple queries
                    reranked_results = await self.reranker.rerank_results(
                        query=query, 
                        results=all_results, 
                        top_k=limit,
                        method="keyword"
                    )
                
                return reranked_results
                
            return []
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    async def generate_response(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        """Generate response with improved context handling"""
        try:
            if not context_docs:
                # Detect language and return appropriate message
                try:
                    lang = detect(query)
                    return (
                        "I couldn't find sufficient information to answer this question."
                        if lang == 'ar' else
                        "I couldn't find any relevant information to answer your question."
                    )
                except Exception as e:
                    logger.error(f"Error detecting language: {e}")
                    return "I couldn't find sufficient information to answer this question."

            # Format context with improved structure
            formatted_context = self._format_context(context_docs)
            
            # Add conversation history if available
            history_context = ""
            if conversation_history:
                history_context = "\nPrevious conversation:\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation_history[-3:]  # Last 3 messages
                )

            # Create prompt with both context and history
            prompt = REACT_PROMPT_TEMPLATE.format(
                context=formatted_context + history_context,
                question=query
            )
            
            # Generate response with timeout
            async with asyncio.timeout(30):  # 30 second timeout
                response = await self.llm.agenerate([prompt])
                
            full_response = response.generations[0][0].text.strip()
            
            # Extract final answer
            answer_parts = full_response.split("Answer:")
            if len(answer_parts) > 1:
                return answer_parts[-1].strip()
            
            # تنظيف وتنسيق النص قبل إرجاعه
            if full_response:
                full_response = self.format_response(full_response)
            
            return full_response

        except asyncio.TimeoutError:
            logger.error("Response generation timed out")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your question."

    async def _generate_search_queries(self, query: str) -> Dict:
        """Generate search queries using enhanced query processing"""
        try:
            # Use the query processor for more sophisticated query processing
            processed_query = await self.query_processor.process_query(query)
            
            # Check if we have meaningful expanded queries from processing
            if processed_query["expanded_queries"]:
                return {
                    "semantic_queries": processed_query["expanded_queries"],
                    "keyword_terms": processed_query["concepts"],
                    "arabic_terms": [] if processed_query["language"] != "ar" else [query],
                    "filters": processed_query["filters"]
                }
            
            # Fall back to original LLM-based approach
            prompt = f"{SEARCH_PROMPT_TEMPLATES['system']}\n\n{SEARCH_PROMPT_TEMPLATES['user'].format(query=query)}"
            
            # Set a reasonable timeout for query generation
            async with asyncio.timeout(5):
                response = await self.llm.agenerate([prompt])
                response_text = response.generations[0][0].text
            
            # Extract JSON from response (handling cases where there's extra text)
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            try:
                search_queries = json.loads(response_text)
                # Validate expected structure
                if not isinstance(search_queries, dict) or "semantic_queries" not in search_queries:
                    raise ValueError("Invalid search query structure")
                    
                return search_queries
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse search queries: {str(e)}")
                # Use enhanced query expansion for better fallback
                return self._get_expanded_queries(query, 
                                                ArabicTextProcessor.contains_arabic(query))
                
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"Error generating search queries: {e}")
            # Use enhanced query expansion for better fallback
            return self._get_expanded_queries(query, ArabicTextProcessor.contains_arabic(query))

    def _get_expanded_queries(self, query: str, is_arabic: bool = False) -> Dict:
        """Get expanded queries with better keyword extraction"""
        # Extract keywords more intelligently
        keywords = []
        
        # More extensive stopword list for better filtering
        stopwords = {
            'and', 'or', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'in', 'for', 'to', 'of', 'with',
            'by', 'as', 'but', 'if', 'from', 'then', 'you', 'have', 'had', 'would', 'could', 'should',
            'were', 'are', 'that', 'this', 'these', 'those', 'there', 'their', 'about'
        }
        
        # Split by spaces and keep non-stopwords
        words = [w.strip('.,?!:;()[]{}"\'-') for w in query.split()]
        keywords = [word for word in words if word.lower() not in stopwords and len(word) > 1]
        
        # Generate query variations
        query_variations = [query]
        
        # Add multi-word combinations for complex queries
        if len(keywords) > 2:
            # Add bigrams and trigrams as variations
            bigrams = [' '.join(keywords[i:i+2]) for i in range(len(keywords)-1)]
            trigrams = [' '.join(keywords[i:i+3]) for i in range(len(keywords)-2)] if len(keywords) > 2 else []
            
            # Add most relevant combinations
            if bigrams:
                query_variations.append(bigrams[0])
            if trigrams:
                query_variations.append(trigrams[0])
            
            # Add first half of the query as a variation (if long enough)
            half_query = ' '.join(words[:len(words)//2])
            query_variations.append(half_query)
        
        return {
            "semantic_queries": query_variations,
            "keyword_terms": keywords,
            "arabic_terms": [query] if is_arabic else [],
            "filters": {}
        }

    async def query_documents(
        self,
        query: str,
        conversation_id: UUID,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Query documents with improved reranking and result processing"""
        try:
            # Make sure RAG service is initialized
            await self.initialize()
            
            logger.info(f"Querying documents for: {query}")
            
            # Process query through advanced query processor 
            processed_query = await self.query_processor.process_query(query)
            logger.info(f"Query type: {processed_query['query_type']}, concepts: {processed_query['concepts']}")
            
            # Perform search with improved hybrid search
            search_results = await self.hybrid_search(
                query=query,
                conversation_id=str(conversation_id),
                limit=limit * 2  # Get more results for better reranking
            )
            
            logger.debug(f"Found {len(search_results)} search results")
            
            if not search_results:
                # Return language-appropriate message
                try:
                    lang = detect(query)
                    message = (
                        "I couldn't find sufficient information to answer this question."
                        if lang == 'ar' else
                        "I couldn't find any relevant information to answer your question."
                    )
                except Exception as e:
                    logger.error(f"Error detecting language: {e}")
                    message = "I couldn't find any relevant information to answer your question."
                    
                return {
                    "response": message,
                    "sources": []
                }

            # Apply semantic reranking for more accurate results
            if len(search_results) > 1:
                try:
                    # For complex queries, use advanced semantic or cross-encoder reranking
                    if processed_query["query_type"] == "complex" and len(query.split()) > 3:
                        # Choose more sophisticated reranking for complex queries
                        reranking_method = "cross_encoder" if len(search_results) <= 10 else "semantic"
                        reranked_results = await self.reranker.rerank_results(
                            query=query, 
                            results=search_results, 
                            top_k=limit,
                            method=reranking_method
                        )
                    else:
                        # For simpler queries, use faster keyword reranking
                        reranked_results = await self.reranker.rerank_results(
                            query=query, 
                            results=search_results, 
                            top_k=limit,
                            method="keyword"
                        )
                    
                    search_results = reranked_results
                except Exception as e:
                    logger.error(f"Error during reranking: {e}")
                    # If reranking fails, use original results
                    if len(search_results) > limit:
                        search_results = search_results[:limit]

            # Generate response with chat history context
            response = await self.generate_response(
                query=query,
                context_docs=search_results
            )
            
            # Return response with source documents
            return {
                "response": response,
                "sources": [
                    {
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {}),
                        "score": doc.get("ensemble_score", 0) or 
                               doc.get("semantic_score", 0) or 
                               doc.get("cross_encoder_score", 0) or
                               doc.get("relevance_score", 0)  # Include most appropriate score
                    }
                    for doc in search_results
                ]
            }

        except Exception as e:
            logger.error(f"Error in document query: {e}", exc_info=True)
            raise

    async def get_document(self, document_id: str, session: Session) -> Document:
        """
        Retrieve a document by its ID.
        """
        document = session.exec(
            select(Document).where(Document.id == document_id)
        ).first()
        
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
            
        return document

    async def check_document_status(self, document_id: UUID) -> Dict:
        """Check document processing status in detail"""
        try:
            doc = await self.get_document(document_id)
            
            status_info = {
                "id": str(document_id),
                "status": doc.current_status,
                "is_searchable": doc.is_searchable,
                "vector_store_ready": bool(doc.vector_ids),
                "chunks_processed": len(doc.chunks),
                "last_event": None
            }

            if doc.processing_events:
                last_event = doc.processing_events[-1]
                status_info["last_event"] = {
                    "status": last_event.status,
                    "message": last_event.message,
                    "timestamp": last_event.created_at
                }

            return status_info

        except Exception as e:
            logger.error(f"Error checking document status: {e}")
            raise

    async def process_attachments(
        self, 
        attachments: List[Attachment], 
        conversation_id: UUID,
        session: Session
    ) -> List[Document]:
        """Process multiple attachments and create documents"""
        documents = []
        
        for attachment in attachments:
            doc = Document(
                title=attachment.filename,
                conversation_id=conversation_id,
                created_by_id=attachment.uploaded_by_id,
                doc_metadata={
                    "original_filename": attachment.filename,
                    "file_type": attachment.file_type,
                    "file_size": attachment.file_size
                }
            )
            
            try:
                processed_doc = await self.process_document(doc, attachment.file_path)
                documents.append(processed_doc)
                session.add(processed_doc)
                
            except Exception as e:
                logger.error(f"Error processing document {attachment.filename}: {e}")
                doc.status = DocumentStatus.FAILED
                doc.current_status = DocumentProcessingStatus.FAILED
                session.add(doc)
                
        session.commit()
        return documents

    def _format_context(self, docs: List[dict]) -> str:
        """Format retrieved documents into context string with improved structure"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Extract content and metadata
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", f"Document {i}")
            title = metadata.get("title", f"Section {i}")
            doc_title = metadata.get("doc_title", "")
            
            # Format with better source information
            header = f"[Document: {doc_title}] [Section: {title}]"
            context_parts.append(f"{header}\n{content}\n")
            
        return "\n---\n".join(context_parts)

    def clean_and_format_text(self, text: str) -> str:
        """Clean and format text"""
        if not text:
            return text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean unwanted marks and symbols
        text = re.sub(r'[.•●■□▪️◾]+(?=\s)', '.', text)  # Standardize bullet points
        text = re.sub(r'["""]', '"', text)  # Standardize quotation marks
        text = re.sub(r'[''`]', "'", text)
        
        # Clean spaces and new lines
        text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)  # Standardize empty lines
        text = re.sub(r' +', ' ', text)  # Remove repeated spaces
        
        # Organize lists
        lines = text.split('\n')
        formatted_lines = []
        list_pattern = re.compile(r'^[\d\-\*\.\s•●■]*\s*(.+)$')
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Format list items
            match = list_pattern.match(line)
            if match:
                content = match.group(1)
                if line.startswith(('•', '●', '■', '*', '-')):
                    formatted_lines.append(f"• {content}")
                elif re.match(r'^\d+[\.\)]', line):
                    num = re.match(r'^\d+', line).group()
                    formatted_lines.append(f"{num}. {content}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        # Recombine the text
        text = '\n'.join(formatted_lines)
        
        # Organize paragraphs
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            if para.strip():
                # Determine if the paragraph is a list
                if any(line.startswith('• ') or re.match(r'^\d+\.', line) 
                      for line in para.split('\n')):
                    # Add space before and after the list
                    formatted_paragraphs.append(f"\n{para}\n")
                else:
                    formatted_paragraphs.append(para)
        
        text = '\n\n'.join(formatted_paragraphs)
        
        # Remove excess spaces at the end
        text = text.strip()
        
        return text

    def format_response(self, response: str) -> str:
        """Format the final response"""
        if not response:
            return response
        
        # Clean the text first
        response = self.clean_and_format_text(response)
        
        # Determine the response pattern
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in response)
        
        # Format numbered and bullet lists
        lines = response.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            # Determine if the line is part of a list
            is_list_item = bool(re.match(r'^[\d\-\*\.\s•●■]', line.strip()))
            
            if is_list_item:
                if not in_list:
                    formatted_lines.append('')  # Space before the list
                    in_list = True
            elif in_list:
                formatted_lines.append('')  # Space after the list
                in_list = False
            
            formatted_lines.append(line)
        
        response = '\n'.join(formatted_lines)
        
        # Format Arabic paragraphs
        if has_arabic:
            # Add space between Arabic paragraphs
            response = re.sub(r'([.؟!])\s*\n', r'\1\n\n', response)
            # Improve Arabic punctuation marks
            response = response.replace('،', '، ').replace('؛', '؛ ')
        
        return response.strip()