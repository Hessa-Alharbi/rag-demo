from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import UUID
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from loguru import logger
from apps.chat.models import Attachment
from core.llm.factory import ModelFactory
from core.llm.prompt_templates import REACT_PROMPT_TEMPLATE, HYBRID_SEARCH_TEMPLATE, SEARCH_PROMPT_TEMPLATES, DIRECT_MATCH_PROMPT_TEMPLATE, LANGUAGE_VALIDATION_TEMPLATE
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
import traceback  # تأكد من استيراد traceback
from core.vector_store.singleton import VectorStoreSingleton
from core.chunking.semantic_chunker import SemanticChunker
from core.search.reranker import QueryResultReranker
from core.search.query_processor import QueryProcessor
from core.language.arabic_utils import ArabicTextProcessor
from langchain_community.document_loaders import PyPDFLoader
from core.db import get_session_context
import os
import traceback
from pydantic import ValidationError
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from apps.users.models import User
# from apps.documents.models import Collection  # Comment out this import to fix the ModuleNotFoundError

class RAGService:
    def __init__(self):
        try:
            self.settings = get_settings()
            self._initialize_lock = asyncio.Lock()
            self._initialized = False
            
            logger.info(f"======= INITIALIZING RAG SERVICE (CONSTRUCTOR) =======")
            logger.info(f"LLM Provider: {self.settings.LLM_PROVIDER}")
            logger.info(f"LLM Model: {self.settings.LLM_MODEL}")
            logger.info(f"Base URL: {self.settings.LLM_BASE_URL}")
            logger.info(f"HF Token available: {bool(self.settings.HF_TOKEN)}")
            
            # تحقق من أن النموذج المستخدم هو yehia-7b-preview-red فقط
            if "yehia-7b-preview-red" not in self.settings.LLM_MODEL.lower():
                error_msg = f"Only yehia-7b-preview-red model is supported. Got: {self.settings.LLM_MODEL}"
                logger.error(f"CONFIGURATION ERROR: {error_msg}")
                # تصحيح الإعداد
                self.settings.LLM_MODEL = "yehia-7b-preview-red"
                logger.info(f"Forced model to: {self.settings.LLM_MODEL}")
            
            # منع استخدام api-inference.huggingface.co
            if "api-inference.huggingface.co" in self.settings.LLM_BASE_URL:
                error_msg = "Invalid endpoint URL or default HuggingFace inference API detected. Must use custom endpoint."
                logger.error(f"CONFIGURATION ERROR: {error_msg}")
                # تصحيح الرابط
                self.settings.LLM_BASE_URL = "https://ijt42iqbf30i3nly.us-east4.gcp.endpoints.huggingface.cloud/v1"
                logger.info(f"Forced endpoint URL to: {self.settings.LLM_BASE_URL}")
            
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
            logger.info("Creating LLM instances using ModelFactory...")
            self.llm = ModelFactory.create_llm()
            self.chat_model = self.llm
            
            # تحقق من النموذج بعد الإنشاء
            model_name = getattr(self.llm, 'model_name', '') or getattr(self.llm, 'repo_id', '') or self.settings.LLM_MODEL
            if "yehia-7b-preview-red" not in str(model_name).lower():
                logger.error(f"WRONG MODEL CREATED: {model_name}")
                # إعادة إنشاء النموذج
                self.llm = ModelFactory.create_llm()
                self.chat_model = ModelFactory.create_llm()
                logger.info("Recreated LLM instances with correct model")
            
            # تم إزالة اختبار LLM المتزامن من constructor لتجنب مشاكل مع event loop
            # سنقوم باختبار LLM في دالة initialize المتزامنة بدلاً من ذلك
            logger.info(f"LLM instances created, testing will be done in async initialize method")
            
            logger.info(f"RAG Service constructor completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RAGService: {str(e)}")

    async def initialize(self):
        """Initialize the RAG service if not already initialized"""
        if not self._initialized:
            async with self._initialize_lock:
                if not self._initialized:
                    # Get settings
                    settings = get_settings()
                    
                    # Log startup information
                    logger.info(f"==== INITIALIZING RAG SERVICE ====")
                    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
                    logger.info(f"LLM Model: {settings.LLM_MODEL}")
                    logger.info(f"Vector Store Provider: {settings.VECTOR_STORE_PROVIDER}")
                    
                    # Validate that only yehia-7b-preview-red model is being used
                    if "yehia-7b-preview-red" not in settings.LLM_MODEL.lower():
                        logger.error(f"CONFIGURATION ERROR: Only yehia-7b-preview-red model is supported. Found: {settings.LLM_MODEL}")
                        raise RuntimeError(f"Unsupported model in configuration: {settings.LLM_MODEL}. Only yehia-7b-preview-red is allowed.")
                                    
                    try:
                        # Create vector store connection first
                        self.vector_store = await VectorStoreSingleton.get_instance()
                        
                        # Initialize services
                        await VectorStoreSingleton.get_instance()
                        
                        # Initialize query processor and reranker
                        self.query_processor = QueryProcessor()
                        await self.query_processor.initialize()
                        
                        self.reranker = QueryResultReranker()
                        await self.reranker.initialize()
                        
                        # Create LLM and verify it's yehia-7b-preview-red
                        logger.info("Creating LLM instances using ModelFactory...")
                        self.llm = ModelFactory.create_llm()
                        self.chat_model = self.llm
                        
                        # اختبار متزامن للتأكد من عمل النموذج
                        logger.info("Directly testing LLM with a simple prompt...")
                        try:
                            # إعادة استخدام نفس طريقة إنشاء النموذج التي تم استخدامها سابقاً
                            # بدلاً من استيراد ModelFactory هنا، نستخدم المستورد على مستوى النطاق العام للوحدة
                            
                            # اختبار النموذج
                            test_response = await asyncio.wait_for(
                                self.llm.agenerate(["اختبار النموذج"], max_tokens=10),
                                timeout=5.0
                            )
                            
                            if test_response and test_response.generations and len(test_response.generations[0]) > 0:
                                logger.info(f"LLM test successful: {test_response.generations[0][0].text}")
                            else:
                                logger.error("LLM test failed - no generations returned")
                                # إعادة إنشاء النموذج
                                self.llm = ModelFactory.create_llm()
                                self.chat_model = ModelFactory.create_llm()
                                logger.info("Recreated LLM instances after test failure")
                        except Exception as e:
                            logger.error(f"Error testing LLM: {e}")
                            logger.error(f"Will try to recreate LLM...")
                            # محاولة إنشاء النموذج مرة أخرى
                            self.llm = ModelFactory.create_llm()
                            self.chat_model = ModelFactory.create_llm()
                        
                        # Initialize chunker
                        self.semantic_chunker = SemanticChunker()
                        
                        # Set initialized flag
                        self._initialized = True
                        logger.info("RAG service initialized successfully with yehia-7b-preview-red model")
                        
                    except Exception as e:
                        logger.error(f"Error initializing RAG service: {e}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        # Re-raise to show the error
                        raise

    async def extract_file_content(self, file_path: str) -> str:
        """Extract text content from different file types"""
        ext = Path(file_path).suffix.lower()
        try:
            if (ext == '.pdf'):
                # استخدام PyPDFLoader مع الحفاظ على النص الأصلي
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                # ضم محتوى الصفحات مع الحفاظ على النص الأصلي
                text = "\n\n".join(page.page_content for page in pages)
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
                
                # Try with original query as a fallback if normalized didn't work
                if len(semantic_results) < 2:
                    original_results = await vector_store.similarity_search(
                        query=query,
                        k=limit * 2,
                        filter={"conversation_id": conversation_id}
                    )
                    
                    # Add results not already in the list
                    existing_ids = {r.get("id") for r in all_results if "id" in r}
                    all_results.extend([r for r in original_results if r.get("id") not in existing_ids])
                
                # Try individual keywords if we still don't have enough results
                if len(all_results) < limit and query_data["keywords"]:
                    # Try each keyword individually for better recall
                    for keyword in query_data["keywords"]:
                        if not keyword or len(keyword) < 3:
                            continue
                            
                        keyword_results = await vector_store.similarity_search(
                            query=keyword,
                            k=limit,
                            filter={"conversation_id": conversation_id}
                        )
                        
                        # Add results not already in the list
                        existing_ids = {r.get("id") for r in all_results if "id" in r}
                        all_results.extend([r for r in keyword_results if r.get("id") not in existing_ids])
                        
                        # Stop if we have enough results
                        if len(all_results) >= limit * 2:
                            break
                            
                # If we still don't have results, try stems
                if len(all_results) < limit and query_data.get("stemmed_keywords"):
                    for stem in query_data["stemmed_keywords"]:
                        if not stem or len(stem) < 3:
                            continue
                            
                        stem_results = await vector_store.similarity_search(
                            query=stem,
                            k=limit,
                            filter={"conversation_id": conversation_id}
                        )
                        
                        # Add results not already in the list
                        existing_ids = {r.get("id") for r in all_results if "id" in r}
                        all_results.extend([r for r in stem_results if r.get("id") not in existing_ids])
                        
                        # Stop if we have enough results
                        if len(all_results) >= limit * 2:
                            break
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
        """توليد استجابة بناءً على السياق المقدم"""
        try:
            logger.info(f"====== GENERATE_RESPONSE START ======")
            logger.info(f"Query: {query}")
            logger.info(f"Context docs count: {len(context_docs)}")
            logger.info(f"History available: {bool(conversation_history)}")
            
            # الحصول على إعدادات النموذج
            settings = get_settings()
            logger.info(f"Using LLM provider: {settings.LLM_PROVIDER}, model: {settings.LLM_MODEL}")
            logger.info(f"Base URL: {settings.LLM_BASE_URL}")
            
            # التحقق من وجود مستندات سياق
            if not context_docs:
                # دائمًا إرجاع رسالة باللغة العربية بغض النظر عن لغة السؤال
                logger.warning("No context documents provided!")
                return "لم أتمكن من العثور على معلومات كافية للإجابة على هذا السؤال."

            # التحقق من أن نموذج اللغة متاح قبل المتابعة - تحقق صارم
            llm_available = False
            llm_error = None
            logger.info(f"---> Verifying LLM availability with test prompt...")
            try:
                # إرسال استعلام بسيط للتحقق من أن النموذج يعمل
                test_prompt = "هل أنت متاح؟"
                logger.info(f"---> Sending test prompt: '{test_prompt}'")
                
                test_response = await asyncio.wait_for(
                    self.llm.agenerate([test_prompt], max_tokens=10),
                    timeout=3.0
                )
                logger.info(f"---> Test response received: {test_response}")
                
                if not test_response.generations or not test_response.generations[0]:
                    logger.error(f"---> Test response has no generations")
                    raise RuntimeError("نموذج اللغة غير متوفر حاليًا")
                
                llm_available = True
                logger.info(f"---> LLM is available and working")
                logger.info(f"---> Test response text: {test_response.generations[0][0].text}")
            except Exception as e:
                llm_error = str(e)
                logger.error(f"نموذج اللغة غير متاح: {e}")
                logger.error(f"---> Error details: {str(e)}")
                logger.error(f"---> Stack trace: {traceback.format_exc()}")
                
                # بدلاً من إعادة رسالة خطأ منسقة، سنقوم برفع استثناء صريح
                error_message = f"نموذج اللغة غير متوفر. الخطأ الأصلي: {llm_error}"
                logger.error(f"!!!! RAISING EXCEPTION: {error_message}")
                raise RuntimeError(error_message)

            # فحص نهائي - التأكد من توفر النموذج
            if not llm_available:
                error_message = "عذرًا، نموذج اللغة غير متوفر حاليًا. يرجى التحقق من إعدادات النظام."
                logger.error(f"!!!! RAISING EXCEPTION: {error_message}")
                raise RuntimeError(error_message)

            # معالجة وتحسين السياق
            logger.info(f"Prioritizing and organizing context documents...")
            optimized_contexts = self._prioritize_and_organize_contexts(context_docs, query)
            logger.info(f"Optimized contexts count: {len(optimized_contexts)}")
            
            formatted_context = self._format_context(optimized_contexts)
            logger.info(f"Formatted context length: {len(formatted_context)} chars")
            logger.debug(f"Formatted context preview: {formatted_context[:200]}...")
            
            # إضافة سجل المحادثة إذا كان متاحًا
            history_context = ""
            if conversation_history:
                logger.info(f"Adding conversation history ({len(conversation_history)} messages)...")
                history_context = "\nالمحادثة السابقة:\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}"
                    for msg in conversation_history[-3:]  # آخر 3 رسائل
                )
                logger.debug(f"History context: {history_context}")
            
            # الحصول على قالب المطالبة المناسب
            logger.info(f"Selecting prompt template...")
            prompt_template = self._select_specialized_prompt(query, context_docs)
            
            # إنشاء مطالبة كاملة مع السياق والتاريخ
            logger.info(f"Creating complete prompt...")
            prompt = prompt_template.format(
                context=formatted_context + history_context,
                question=query
            )
            
            # إضافة تعليمات صريحة للإجابة باللغة العربية فقط
            logger.info(f"Adding Arabic-only instruction...")
            arabic_instruction = "هام: يجب أن تكون إجابتك باللغة العربية فقط مهما كانت لغة السؤال.\n\n"
            prompt = arabic_instruction + prompt
            
            logger.info(f"Final prompt length: {len(prompt)} chars")
            logger.debug(f"Final prompt preview: {prompt[:200]}...")
            
            # توليد الاستجابة مع مهلة زمنية محددة والإعدادات المطلوبة
            logger.info(f"Generating response with LLM...")
            try:
                async with asyncio.timeout(30):  # مهلة 30 ثانية
                    logger.info(f"Calling LLM.agenerate with max_tokens=500...")
                    response = await self.llm.agenerate(
                        [prompt],
                        max_tokens=500,
                        stream=True
                    )
                    logger.info(f"LLM response received: {response}")
                    
                    if not response.generations:
                        logger.error("Response has no generations!")
                        return "عذرًا، حدث خطأ أثناء توليد الإجابة. يرجى المحاولة مرة أخرى."
                        
                    raw_response = response.generations[0][0].text.strip()
                    logger.info(f"Raw response length: {len(raw_response)} chars")
                    logger.debug(f"Raw response: {raw_response}")
                    
                    # التأكد من أن الإجابة بالعربية
                    if not self._is_arabic_text(raw_response):
                        logger.warning("Response was not in Arabic, regenerating")
                        # إذا لم تكن الإجابة بالعربية، نعيد توليدها مع تعليمات أكثر وضوحًا
                        enhanced_prompt = prompt + "\n\nيجب أن تكون إجابتك باللغة العربية فقط. لا تستخدم أي لغة أخرى."
                        logger.info("Calling LLM again with enhanced Arabic instructions...")
                        response = await self.llm.agenerate(
                            [enhanced_prompt],
                            max_tokens=500,
                            stream=True
                        )
                        raw_response = response.generations[0][0].text.strip()
                        logger.info(f"Regenerated raw response length: {len(raw_response)} chars")
                        logger.debug(f"Regenerated raw response: {raw_response}")
                    
                    # استخراج الإجابة النهائية إذا لزم الأمر
                    if "Answer:" in raw_response:
                        logger.info("Extracting answer part after 'Answer:'")
                        answer_parts = raw_response.split("Answer:")
                        raw_response = answer_parts[-1].strip()
                    
                    if "الإجابة:" in raw_response:
                        logger.info("Extracting answer part after 'الإجابة:'")
                        answer_parts = raw_response.split("الإجابة:")
                        raw_response = answer_parts[-1].strip()
                    
                    # تطبيق معالجة ما بعد التوليد
                    logger.info("Post-processing response...")
                    processed_response = self._post_process_response(raw_response, query)
                    
                    # التحقق من جودة الاستجابة
                    logger.info("Validating response quality...")
                    if not self._validate_response_quality(processed_response, query):
                        logger.info("Response quality insufficient, attempting refinement")
                        refined_response = await self._generate_refined_response(query, context_docs)
                        logger.info(f"Refined response generated: {refined_response[:100]}...")
                        return refined_response
                    
                    logger.info(f"====== GENERATE_RESPONSE COMPLETE ======")
                    logger.info(f"Final response length: {len(processed_response)} chars")
                    logger.debug(f"Final response: {processed_response}")
                    return processed_response
            except asyncio.TimeoutError:
                logger.error("Response generation timed out")
                error_message = "أعتذر، لقد استغرقت العملية وقتًا طويلًا. يرجى المحاولة مرة أخرى."
                logger.error(f"!!!! RAISING EXCEPTION: {error_message}")
                raise RuntimeError(error_message)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            # بدلاً من إرجاع رسالة خطأ، نقوم برفع الاستثناء للتعامل معه على مستوى أعلى
            raise RuntimeError(f"Error generating response: {str(e)}")

    def _is_arabic_text(self, text: str) -> bool:
        """التحقق من أن النص عربي بنسبة عالية"""
        if not text:
            return False
        
        # عد الأحرف العربية
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        
        # عد أحرف التشكيل
        diacritics = sum(1 for c in text if '\u064B' <= c <= '\u065F')
        
        # عد الحروف اللاتينية
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        
        # طول النص بدون المسافات
        non_space_length = sum(1 for c in text if not c.isspace())
        
        if non_space_length == 0:
            return False
        
        # نسبة الأحرف العربية
        arabic_ratio = (arabic_chars - diacritics) / non_space_length
        
        # نسبة الحروف اللاتينية
        latin_ratio = latin_chars / non_space_length
        
        # اختبار صارم: يجب أن تكون النسبة العربية عالية والنسبة اللاتينية منخفضة
        return arabic_ratio > 0.75 and latin_ratio < 0.15

    def _select_specialized_prompt(self, query: str, context_docs: List[Dict[str, Any]]):
        """Select the most appropriate prompt template based on query type and language"""
        from core.llm.prompt_templates import REACT_PROMPT_TEMPLATE
        
        # استخدام قالب REACT_PROMPT_TEMPLATE الافتراضي لجميع الاستعلامات
        logger.info("Using default REACT_PROMPT_TEMPLATE - This is the only template currently in use")
        
        # تسجيل محتوى القالب بطريقة آمنة
        try:
            template_content = str(REACT_PROMPT_TEMPLATE)
            logger.debug(f"REACT_PROMPT_TEMPLATE preview: {template_content[:100] if len(template_content) > 100 else template_content}...")
        except Exception as e:
            logger.error(f"Error accessing template content: {e}")
        
        # إضافة تسجيل لمحتوى المطالبة الكامل لتتبع أفضل
        try:
            with open("prompt_template_used.log", "w", encoding="utf-8") as f:
                f.write(f"Template used for query: {query}\n\n")
                f.write(str(REACT_PROMPT_TEMPLATE))
        except Exception as e:
            logger.error(f"Error writing template to log: {e}")
        
        return REACT_PROMPT_TEMPLATE

    def _prioritize_and_organize_contexts(self, context_docs: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Prioritize and organize context documents for better response generation"""
        # Clone the context docs to avoid modifying original
        docs = context_docs.copy()
        
        # First, identify query language and key terms
        is_arabic = any(c for c in query if '\u0600' <= c <= '\u06FF')
        
        # Extract key terms from query for relevance calculation
        if is_arabic:
            query_data = ArabicTextProcessor.preprocess_arabic_query(query)
            key_terms = [query_data["normalized"]] + query_data["keywords"]
        else:
            # Simple English tokenization
            key_terms = [term.lower() for term in re.findall(r'\b\w+\b', query)]
            # Remove English stopwords
            stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'is', 'are', 'was', 'were'}
            key_terms = [term for term in key_terms if term not in stopwords]
        
        # Score documents by term overlap and position of key terms
        scored_docs = []
        for doc in docs:
            content = doc.get("content", "").lower()
            
            # Calculate initial score based on term overlap
            term_score = sum(10 for term in key_terms if term.lower() in content)
            
            # Bonus for terms appearing early in the content
            for term in key_terms:
                term_lower = term.lower()
                if term_lower in content:
                    # Position weight: earlier = better
                    position = content.find(term_lower)
                    position_score = max(10 - (position / 100), 0) if position >= 0 else 0
                    term_score += position_score
            
            # Bonus for shorter, more focused content
            length_factor = min(1.0, 500 / max(len(content), 1))
            term_score *= (0.5 + 0.5 * length_factor)
            
            # Use existing semantic score if available
            semantic_score = doc.get("semantic_score", 0) or doc.get("score", 0)
            
            # Combine scores with preference for semantic matching
            final_score = (semantic_score * 0.7) + (term_score * 0.3)
            
            scored_docs.append((doc, final_score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reorganized docs
        organized_docs = [doc for doc, _ in scored_docs]
        
        # Optionally, truncate very long documents to focus on most relevant parts
        for i in range(len(organized_docs)):
            content = organized_docs[i].get("content", "")
            if len(content) > 1000:  # If content is very long
                # Find positions of key terms
                positions = []
                for term in key_terms:
                    term_lower = term.lower()
                    pos = content.lower().find(term_lower)
                    if pos >= 0:
                        positions.append(pos)
                
                if positions:
                    # Calculate a central position around which to extract content
                    center_pos = sum(positions) // len(positions)
                    
                    # Extract a window around this position
                    start_pos = max(0, center_pos - 400)
                    end_pos = min(len(content), center_pos + 600)
                    
                    # Get the truncated content with context indication
                    truncated = content[start_pos:end_pos]
                    if start_pos > 0:
                        truncated = "... " + truncated
                    if end_pos < len(content):
                        truncated = truncated + " ..."
                    
                    # Update the document content
                    organized_docs[i]["content"] = truncated
        
        return organized_docs

    def _post_process_response(self, response: str, query: str) -> str:
        """تنظيف وتحسين بسيط للاستجابة"""
        if not response:
            return response
        
        # تنظيف الاستجابة
        response = response.strip()
        
        # إزالة علامات الاقتباس والمراجع
        response = re.sub(r'\[\d+\]', '', response)
        response = re.sub(r'\(Source: [^)]+\)', '', response)
        response = re.sub(r'\(المصدر: [^)]+\)', '', response)
        
        # تنظيف المسافات والتنسيق
        response = re.sub(r'\s{2,}', ' ', response)
        response = re.sub(r'[.،,؛;:]{2,}', '.', response)
        
        # التأكد من وجود علامة ترقيم في النهاية
        if not response.endswith(('.', '!', '?', '؟', '.')):
            response += '.'
        
        return response

    def _validate_response_quality(self, response: str, query: str) -> bool:
        """تحقق بسيط من جودة الاستجابة"""
        # التحقق من وجود استجابة
        if not response:
            return False
            
        # التحقق من الحد الأدنى للطول
        if len(response) < 10:
            return False
            
        # التحقق من عدم وجود عبارات الخطأ الشائعة
        error_phrases = [
            "I apologize",
            "I cannot provide",
            "I don't have access",
            "لا أستطيع",
            "لا يمكنني",
            "المعلومات غير متوفرة"
        ]
        
        for phrase in error_phrases:
            if phrase in response:
                return False
                
        return True

    async def _generate_refined_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """يولد استجابة محسنة عندما تكون جودة الاستجابة الأولية غير كافية"""
        try:
            # استخدام أكثر 3 مستندات صلة
            key_docs = context_docs[:min(3, len(context_docs))]
            
            # تنسيق السياق من هذه المستندات المهمة فقط
            focused_context = self._format_context(key_docs)
            
            # استخدام قالب المطابقة المباشرة للحصول على إجابة أكثر دقة
            from core.llm.prompt_templates import DIRECT_MATCH_PROMPT_TEMPLATE
            
            prompt = DIRECT_MATCH_PROMPT_TEMPLATE.format(
                context=focused_context,
                question=query
            )
            
            # إضافة تعليمات صريحة للإجابة باللغة العربية فقط
            prompt = "هام: يجب أن تكون إجابتك باللغة العربية فقط مهما كانت لغة السؤال.\n\n" + prompt
            
            # توليد الاستجابة مع مهلة زمنية
            async with asyncio.timeout(30):
                response = await self.llm.agenerate(
                    [prompt],
                    max_tokens=500,
                    stream=True
                )
                
            reasoned_response = response.generations[0][0].text.strip()
            
            # استخراج الإجابة النهائية إذا كانت بصيغة معينة
            if "Answer:" in reasoned_response:
                answer_parts = reasoned_response.split("Answer:")
                final_answer = answer_parts[-1].strip()
            elif "الإجابة:" in reasoned_response:
                answer_parts = reasoned_response.split("الإجابة:")
                final_answer = answer_parts[-1].strip()
            else:
                final_answer = reasoned_response
            
            # التأكد من أن الإجابة بالعربية
            if not self._is_arabic_text(final_answer):
                logger.warning("Refined response was not in Arabic, forcing Arabic")
                return "لم أتمكن من الحصول على إجابة واضحة. يرجى إعادة صياغة السؤال بطريقة أخرى."
            
            # تطبيق معالجة ما بعد التوليد
            return self._post_process_response(final_answer, query)
            
        except Exception as e:
            logger.error(f"Error in refined response generation: {e}")
            # تغيير السلوك لمنع استخدام الاستخراج المباشر عندما يكون النموذج غير متاح
            return "عذرًا، نموذج اللغة غير متوفر حاليًا. يرجى التحقق من إعدادات النظام."

    def _extract_direct_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """استخراج إجابة مباشرة من أكثر جزء ذي صلة - معطلة الآن وترجع رسالة خطأ فقط"""
        # تم تعطيل الاستخراج المباشر من المستندات في حالة عدم توفر نموذج اللغة
        return "عذرًا، النموذج اللغوي غير متوفر حاليًا. يرجى التحقق من إعدادات النظام."

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
            
            logger.info(f"====== QUERY DOCUMENTS START ======")
            logger.info(f"Query: {query}")
            logger.info(f"Conversation ID: {conversation_id}")
            
            # الكشف عن تشغيل النموذج - التحقق من إعداد REQUIRE_LLM في ملف الإعدادات
            settings = get_settings()
            require_llm = getattr(settings, "REQUIRE_LLM", True)  # افتراضياً نطلب أن يكون النموذج متاحًا
            logger.info(f"REQUIRE_LLM setting: {require_llm}")
            
            # اختبار محسن للتحقق من أن نموذج اللغة يعمل
            llm_available = False
            llm_error = None
            logger.info(f"Testing LLM availability...")
            try:
                # توثيق بداية فحص النموذج بسجل واضح
                logger.info(f"---> التحقق من توفر نموذج اللغة، الإعداد REQUIRE_LLM={require_llm}")
                logger.info(f"---> إعدادات النموذج: {settings.LLM_PROVIDER}/{settings.LLM_MODEL}")
                logger.info(f"---> رابط النموذج: {settings.LLM_BASE_URL}")

                # تنفيذ اختبار صارم ومباشر
                test_prompt = "اختبار التوفر فقط"
                logger.info(f"---> إرسال استعلام اختبار: '{test_prompt}'")
                
                test_response = await asyncio.wait_for(
                    self.llm.agenerate([test_prompt], max_tokens=5),
                    timeout=5.0  # زيادة مهلة الزمن
                )
                
                logger.info(f"---> استلام الاستجابة: {test_response}")
                
                # فحص متعمق للاستجابة
                if (test_response and 
                    hasattr(test_response, 'generations') and 
                    test_response.generations and 
                    test_response.generations[0] and 
                    len(test_response.generations[0]) > 0 and
                    hasattr(test_response.generations[0][0], 'text') and
                    test_response.generations[0][0].text):
                    llm_available = True
                    logger.info(f"---> تم التأكد من توفر نموذج اللغة بنجاح")
                    logger.info(f"---> نص الاستجابة: {test_response.generations[0][0].text}")
                else:
                    llm_error = "استجابة فارغة أو غير صالحة من النموذج"
                    logger.error(f"---> النموذج غير متاح: {llm_error}")
                    logger.error(f"---> تفاصيل الاستجابة: {test_response}")
                    
            except asyncio.TimeoutError as e:
                llm_error = f"انتهت مهلة الزمن: {str(e)}"
                logger.error(f"---> انتهت مهلة الاتصال بالنموذج: {e}")
                logger.error(f"---> تفاصيل الخطأ: {str(e)}")
            except Exception as e:
                llm_error = f"خطأ عام: {str(e)}"
                logger.error(f"---> النموذج غير متاح (خطأ): {e}")
                logger.error(f"---> تفاصيل الخطأ: {str(e)}")
                logger.error(f"---> مسار التتبع: {traceback.format_exc()}")
            
            # قرار حاسم عند عدم توفر النموذج - إضافة رسالة فخ مميزة للكشف عن مصدر الاستجابات
            if not llm_available:
                logger.warning(f"===> منع الاستجابة: النموذج غير متوفر، REQUIRE_LLM={require_llm}")
                
                # تعديل الرسالة لتكون فريدة ومميزة جداً بحيث يسهل تتبعها في المخرجات
                error_message = f"MODIFIED-ERROR-CODE-ABC1234: النموذج اللغوي غير متوفر! تم تعديل الكود للكشف عن مصدر الاستجابات. [رمز الخطأ: TRACKING-ID-567890]"
                
                logger.error(f"!!!! RETURNING MODIFIED UNIQUE ERROR: {error_message}")
                
                # إرجاع رسالة خطأ دائمًا بغض النظر عن قيمة REQUIRE_LLM
                return {
                    "response": error_message,
                    "sources": [],
                    "error": "llm_unavailable_tracked",
                    "llm_error": llm_error
                }
            
            logger.info(f"Continuing processing with LLM available")
            
            # Process query through advanced query processor 
            processed_query = await self.query_processor.process_query(query)
            logger.info(f"Query type: {processed_query['query_type']}, concepts: {processed_query['concepts']}")
            
            # Perform search with improved hybrid search
            logger.info(f"Performing hybrid search...")
            search_results = await self.hybrid_search(
                query=query,
                conversation_id=str(conversation_id),
                limit=limit * 3  # Get more results for better reranking
            )
            
            logger.info(f"Found {len(search_results)} search results")
            
            if not search_results:
                logger.warning("No search results found")
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
                
                logger.info(f"Returning no results message: {message}")    
                return {
                    "response": message,
                    "sources": []
                }

            # Apply semantic reranking for more accurate results
            if len(search_results) > 1:
                logger.info(f"Reranking {len(search_results)} results...")
                try:
                    # For complex queries, use advanced semantic or cross-encoder reranking
                    if processed_query["query_type"] == "complex" and len(query.split()) > 3:
                        # Choose more sophisticated reranking for complex queries
                        reranking_method = "cross_encoder" if len(search_results) <= 10 else "semantic"
                        logger.info(f"Using {reranking_method} reranking for complex query")
                        reranked_results = await self.reranker.rerank_results(
                            query=query, 
                            results=search_results, 
                            top_k=limit,
                            method=reranking_method
                        )
                    else:
                        # For simpler queries, use faster keyword reranking
                        logger.info(f"Using keyword reranking for simple query")
                        reranked_results = await self.reranker.rerank_results(
                            query=query, 
                            results=search_results, 
                            top_k=limit,
                            method="keyword"
                        )
                    
                    search_results = reranked_results
                    logger.info(f"Reranking complete, {len(search_results)} results after reranking")
                except Exception as e:
                    logger.error(f"Error during reranking: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    # If reranking fails, use original results
                    if len(search_results) > limit:
                        search_results = search_results[:limit]
                        logger.info(f"Using top {limit} results without reranking")

            # تحديد مصدر الإجابة بناءً على توفر النموذج
            if llm_available:
                # استخدام النموذج لإنشاء إجابة
                logger.info(f"Calling generate_response with {len(search_results)} context documents...")
                response = await self.generate_response(
                    query=query,
                    context_docs=search_results
                )
                logger.info(f"Response generated: {response[:100]}...")
            else:
                # إذا كان النموذج غير متاح، نعيد رسالة خطأ واضحة ولا نستخدم استخراج النص المباشر
                logger.warning("LLM unavailable, returning clear error message")
                error_message = f"عذرًا، النموذج اللغوي غير متوفر حاليًا. يرجى التحقق من إعدادات النظام.\n\nالخطأ: {llm_error}"
                return {
                    "response": error_message,
                    "sources": [],
                    "error": "llm_unavailable_tracked",
                    "llm_error": llm_error
                }
            
            # Enhance source information - ensure file name is included
            logger.info(f"Enhancing source information for {len(search_results)} results")
            enhanced_sources = []
            for doc in search_results:
                metadata = doc.get("metadata", {})
                
                # Extract filename from source path if available
                source_path = metadata.get("source", "")
                file_name = "Unknown Source"
                
                if source_path:
                    try:
                        file_name = os.path.basename(source_path)
                    except (TypeError, AttributeError):
                        # If source_path isn't a valid path, use it directly
                        file_name = str(source_path)
                elif metadata.get("doc_title"):
                    file_name = metadata.get("doc_title")
                
                # Create enhanced source record
                enhanced_source = {
                    "content": doc.get("content", ""),
                    "metadata": {
                        **metadata,
                        "file_name": file_name,  # Add explicit file name
                        "title": metadata.get("title", "Unknown Section")
                    },
                    "score": doc.get("ensemble_score", 0) or 
                           doc.get("semantic_score", 0) or 
                           doc.get("cross_encoder_score", 0) or
                           doc.get("relevance_score", 0)  # Include most appropriate score
                }
                enhanced_sources.append(enhanced_source)
            
            # Return response with enhanced source documents
            logger.info(f"====== QUERY DOCUMENTS COMPLETE ======")
            result = {
                "response": response,
                "sources": enhanced_sources,
                "using_llm": llm_available  # إضافة علامة توضح ما إذا تم استخدام النموذج
            }
            logger.info(f"Returning result with response and {len(enhanced_sources)} sources")
            return result

        except Exception as e:
            logger.error(f"Error in document query: {e}", exc_info=True)
            logger.error(f"Stack trace: {traceback.format_exc()}")
            error_message = "عذرًا، حدث خطأ أثناء معالجة الاستعلام. يرجى المحاولة مرة أخرى لاحقًا."
            logger.error(f"!!!! RETURNING ERROR: {error_message}")
            return {
                "response": error_message,
                "sources": [],
                "error": str(e)
            }

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
        """تنسيق وثائق السياق بطريقة منظمة ومفيدة"""
        if not docs:
            return ""
        
        formatted_sections = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue
            
            # تضمين عنوان للقسم (استخدام العنوان من البيانات الوصفية إذا كان متاحًا)
            metadata = doc.get("metadata", {})
            title = metadata.get("title", "") or f"مستند {i}"
            source = metadata.get("file_name", "") or metadata.get("source", "")
            
            # إضافة معلومات حول المصدر
            section_header = f"== {title} =="
            if source:
                section_header += f" (المصدر: {source})"
            
            # تنسيق النص مع فصل واضح بين الأقسام
            formatted_section = f"{section_header}\n{content}\n"
            formatted_sections.append(formatted_section)
        
        # الجمع بين الأقسام مع فواصل واضحة
        return "\n---\n".join(formatted_sections)

    def clean_and_format_text(self, text: str) -> str:
        """Clean and format text"""
        if not text:
            return text
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # للنصوص العربية، نحافظ على النص كما هو دون معالجة
        if ArabicTextProcessor.contains_arabic(text):
            # نزيل فقط المسافات الزائدة
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        # Clean unwanted marks and symbols (للنصوص غير العربية فقط)
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
        """تنسيق بسيط للاستجابة"""
        if not response:
            return response
            
        # تنظيف النص
        response = response.strip()
        response = re.sub(r'\n{3,}', '\n\n', response)  # تنظيف الأسطر المتكررة
        response = re.sub(r' {2,}', ' ', response)  # تنظيف المسافات المتكررة
        
        # إزالة علامات الاقتباس
        response = re.sub(r'\[\d+\]', '', response)
        
        return response

    async def index_document(self, file_path: str, conversation_id: str, metadata: Dict[str, Any] = None) -> Document:
        """Index a document for a conversation and add it to the vector store"""
        from sqlmodel import Session
        from core.db import get_session_context
        from uuid import uuid4
        import os
        
        async with get_session_context() as session:
            try:
                # Get the conversation to find the user ID
                from sqlmodel import select
                from apps.chat.models import Conversation
                
                # Query the conversation to get the user ID
                conversation = session.exec(
                    select(Conversation).where(Conversation.id == UUID(conversation_id))
                ).first()
                
                if not conversation:
                    raise ValueError(f"Conversation with ID {conversation_id} not found")
                
                # Create document record with the conversation owner's ID
                doc = Document(
                    id=uuid4(),
                    conversation_id=UUID(conversation_id),
                    title=os.path.basename(file_path),
                    source=file_path,
                    type=Path(file_path).suffix.lower().replace('.', ''),
                    status=DocumentStatus.PROCESSING,
                    is_searchable=False,
                    document_metadata=metadata or {},
                    created_by_id=conversation.created_by_id  # Add the created_by_id field
                )
                
                session.add(doc)
                session.commit()
                session.refresh(doc)
                
                # Process the document
                processed_doc = await self.process_document(doc, file_path)
                
                # Update the document in the database
                session.add(processed_doc)
                session.commit()
                
                return processed_doc
                
            except Exception as e:
                logger.error(f"Error indexing document: {e}")
                session.rollback()
                raise

    def _initialize_qa_chain(self):
        """Initialize QA chain for generating answers."""
        try:
            system_template = """أنت مساعد يقوم بالإجابة على الأسئلة بناءً على المعلومات المحددة المقدمة. قم بالإجابة فقط باستخدام المحتوى المقدم دون إختراع معلومات.

المعلومات المتاحة:

{context}

قم بالإجابة باللغة العربية بطريقة مفيدة ومهذبة ودقيقة. إذا لم تعرف الإجابة أو لم تكن المعلومات متوفرة في السياق المقدم، فقط اعترف بذلك بدلاً من اختراع المعلومات.
"""
            system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
            
            human_template = "{question}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
            
            self.qa_chain = load_qa_chain(
                self.chat_model,
                chain_type="stuff",
                prompt=chat_prompt,
            )
            
            logger.info("QA chain initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QA chain: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def process_query(
        self,
        query: str,
        document_ids: List[str] = None,
        options: Dict[str, Any] = None,
    ):
        """Process a query and return the answer with relevant context."""
        try:
            options = options or {}
            
            # Query preprocessing
            self.query_processor.process(query)
            
            # Retrieve documents
            docs = self._retrieve_documents(query, document_ids, options)
            
            if not docs:
                return {
                    "answer": "لم أجد أي معلومات ذات صلة بسؤالك.",
                    "context": [],
                    "metadata": {"status": "no_documents_found"},
                }
            
            # Rerank documents
            if self.reranker and len(docs) > 1:
                docs = self.reranker.rerank(query, docs)
            
            # Generate answer
            answer = self._generate_answer(query, docs)
            
            # Prepare context for response
            context = []
            for i, doc in enumerate(docs[:self.settings.RAG_MAX_DOCUMENTS_IN_RESPONSE]):
                context.append({
                    "document_id": doc.metadata.get("document_id", f"doc_{i}"),
                    "title": doc.metadata.get("title", f"Document {i+1}"),
                    "content": doc.page_content[:1000],
                    "score": doc.metadata.get("score", 0),
                })
            
            # Return result
            return {
                "answer": answer,
                "context": context,
                "metadata": {
                    "model": getattr(self.chat_model, "model_name", "unknown"),
                    "document_count": len(docs),
                },
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to process query: {str(e)}")

    def _retrieve_documents(
        self,
        query: str,
        document_ids: List[str] = None,
        options: Dict[str, Any] = None,
    ) -> List[Document]:
        """Retrieve relevant documents for the query."""
        # Simple document retrieval logic
        # In a real implementation, this would search a vector database
        return self.query_processor.search(query, document_ids=document_ids, **options)

    def _generate_answer(self, query: str, docs: List[Document]) -> str:
        """Generate an answer for the query using the retrieved documents."""
        if not docs:
            return "لم أجد أي معلومات ذات صلة بسؤالك."
        
        try:
            result = self.qa_chain(
                {"input_documents": docs, "question": query, "context": "\n\n".join([doc.page_content for doc in docs])}
            )
            return result.get("output_text", "عذراً، لم أتمكن من إنشاء إجابة.")
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return "عذراً، حدث خطأ أثناء توليد الإجابة."