from typing import List, Dict, Any, Optional
from uuid import UUID
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from langchain_community.retrievers import BM25Retriever
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
from core.vector_store.manager import VectorStoreManager
from core.queue.manager import AsyncTaskQueue
from sqlmodel import Session, select
from core.db import get_session
from langdetect import detect
import arabic_reshaper
from bidi.algorithm import get_display
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import asyncio
from core.vector_store.singleton import VectorStoreSingleton

class RAGService:
    def __init__(self):
        try:
            self.settings = get_settings()
            self._initialize_lock = asyncio.Lock()
            self._initialized = False
            
            # Initialize text splitter with improved settings
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize LLM
            self.llm = ModelFactory.create_llm()
            # self.chat_model = ChatOpenAI(
            #     model_name=self.settings.OPENAI_MODEL_NAME,
            #     temperature=0.0
            # )
            self.chat_model = ModelFactory.create_llm()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RAGService: {str(e)}")

    async def initialize(self):
        """Initialize global vector store"""
        if not self._initialized:
            async with self._initialize_lock:
                if not self._initialized:
                    await VectorStoreSingleton.get_instance()
                    self._initialized = True

    async def extract_file_content(self, file_path: str) -> str:
        """Extract text content from different file types"""
        ext = Path(file_path).suffix.lower()
        try:
            if (ext == '.pdf'):
                text = ""
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text += page.get_text()
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
        """Process document and add to vector store"""
        try:
            # Get conversation-specific vector store
            vector_store = await VectorStoreSingleton.get_conversation_store(str(doc.conversation_id))
            
            # Extract and process content
            content = await self.extract_file_content(file_path)
            doc.content = content
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Prepare chunks for vector store
            texts = []
            metadatas = []
            db_chunks = []
            
            for i, chunk_text in enumerate(chunks):
                chunk = Chunk(
                    document_id=doc.id,
                    content=chunk_text,
                    chunk_metadata={
                        "index": i,
                        "source": file_path
                    }
                )
                db_chunks.append(chunk)
                texts.append(chunk_text)
                metadatas.append({
                    "document_id": str(doc.id),
                    "conversation_id": str(doc.conversation_id),
                    "chunk_index": i,
                    "source": file_path
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
            results = await self.vector_store.similarity_search(
                query,
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
        """Improved hybrid search with conversation isolation"""
        try:
            # Get conversation-specific vector store
            vector_store = await VectorStoreSingleton.get_conversation_store(conversation_id)
            
            if not query.strip():
                return []
                
            # Generate search queries
            search_queries = await self._generate_search_queries(query)
            
            # Perform semantic search
            semantic_results = await vector_store.similarity_search(
                query=search_queries["semantic_queries"][0],
                k=limit,
                filter={"conversation_id": conversation_id}
            )
            
            if not semantic_results:
                # Try keyword search as fallback
                keyword_query = " ".join(search_queries["keyword_terms"])
                semantic_results = await vector_store.similarity_search(
                    query=keyword_query,
                    k=limit,
                    filter={"conversation_id": conversation_id}
                )
            
            return semantic_results
            
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
                        "لم أتمكن من العثور على معلومات كافية للإجابة على هذا السؤال"
                        if lang == 'ar' else
                        "I cannot find sufficient information to answer this question."
                    )
                except:
                    return "I cannot find sufficient information to answer this question."

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
            
            return full_response

        except asyncio.TimeoutError:
            logger.error("Response generation timed out")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your question."

    async def _generate_search_queries(self, query: str) -> Dict:
        """Generate search queries using LLM"""
        try:
            # Use direct prompt format instead of chat messages
            prompt = f"{SEARCH_PROMPT_TEMPLATES['system']}\n\n{SEARCH_PROMPT_TEMPLATES['user'].format(query=query)}"
            
            response = await self.llm.agenerate([prompt])
            response_text = response.generations[0][0].text
            
            try:
                search_queries = json.loads(response_text)
                return search_queries
            except json.JSONDecodeError:
                logger.error("Failed to parse search queries")
                # Detect language for fallback
                is_arabic = False
                try:
                    is_arabic = detect(query) == 'ar'
                except:
                    pass
                return self._get_default_queries(query, is_arabic)
                
        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            # Detect language for fallback
            is_arabic = False
            try:
                is_arabic = detect(query) == 'ar'
            except:
                pass
            return self._get_default_queries(query, is_arabic)

    def _get_default_queries(self, query: str, is_arabic: bool = False) -> Dict:
        """Get default queries with Arabic support"""
        return {
            "semantic_queries": [query],
            "keyword_terms": query.split(),
            "arabic_terms": [query] if is_arabic else [],
            "filters": {}
        }

    def _merge_search_results(
        self,
        semantic_results: List[dict],
        keyword_results: List[dict],
        limit: int
    ) -> List[dict]:
        """Merge and deduplicate search results"""
        # Implement custom merging logic
        seen_ids = set()
        merged = []
        
        for result in semantic_results + keyword_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                merged.append(result)
                if len(merged) >= limit:
                    break
                    
        return merged

    def _format_context(self, docs: List[dict]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Extract content and metadata
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", f"Document {i}")
            
            # Format with source information
            context_parts.append(f"[Source: {source}]\n{content}\n")
            
        return "\n---\n".join(context_parts)

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

    async def query_documents(
        self,
        query: str,
        conversation_id: UUID,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Query documents with improved error handling"""
        try:
            # Initialize if needed
            if not self.vector_store._initialized:
                await self.vector_store.initialize()
            
            logger.info(f"Querying documents for: {query}")
            
            # Perform search
            search_results = await self.hybrid_search(
                query=query,
                conversation_id=str(conversation_id),
                limit=limit
            )
            
            logger.debug(f"Found {len(search_results)} search results")
            
            if not search_results:
                # Return language-appropriate message
                try:
                    lang = detect(query)
                    message = (
                        "لم أتمكن من العثور على معلومات كافية للإجابة على هذا السؤال"
                        if lang == 'ar' else
                        "I couldn't find any relevant information to answer your question."
                    )
                except:
                    message = "I couldn't find any relevant information to answer your question."
                    
                return {
                    "response": message,
                    "sources": []
                }

            # Generate response
            response = await self.generate_response(
                query=query,
                context_docs=search_results
            )
            
            return {
                "response": response,
                "sources": [
                    {
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in search_results
                ]
            }

        except Exception as e:
            logger.error(f"Error in document query: {e}", exc_info=True)
            raise