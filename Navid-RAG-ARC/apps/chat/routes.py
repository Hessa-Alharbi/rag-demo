import sys
import os
import json
from uuid import UUID
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query, Body, status
from sqlmodel import Session, select, or_, and_
from datetime import datetime, timedelta
from loguru import logger
import asyncio
from apps.chat.models import Conversation, Message, Attachment, MessageRole, MessageStatus
from apps.chat.schemas import ConversationCreate, ConversationRead, MessageCreate, MessageRead, AttachmentRead
from apps.chat.utils import FileHandler
from apps.users.models import User
from core.db import get_session
from apps.auth.routes import get_current_user
from core.errors import APIError
from apps.rag.services import RAGService
from apps.rag.models import Document, DocumentStatus, Chunk
from core.config import get_settings
import traceback


router = APIRouter()
rag_service = RAGService()

async def verify_conversation_access(
    conversation_id: UUID,
    user_id: UUID,
    session: Session
) -> Conversation:
    """Verify user has access to conversation and return it"""
    conversation = session.exec(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.created_by_id == user_id
        )
    ).first()
    
    if not conversation:
        raise APIError(
            status_code=status.HTTP_404_NOT_FOUND,
            error="conversation_not_found",
            message="Conversation not found or you don't have permission to access it"
        )
    
    return conversation

async def create_user_message(
    content: str,
    conversation_id: UUID,
    user_id: UUID,
    session: Session
) -> Message:
    """Create a user message"""
    message = Message(
        content=content,
        conversation_id=conversation_id,
        user_id=user_id,
        role=MessageRole.USER,
        status=MessageStatus.COMPLETED
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    return message

async def create_assistant_message(
    conversation_id: UUID,
    user_id: UUID,
    session: Session
) -> Message:
    """Create a pending assistant message"""
    message = Message(
        content="",
        conversation_id=conversation_id,
        user_id=user_id,
        role=MessageRole.ASSISTANT,
        status=MessageStatus.PROCESSING
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    return message

def get_conversation_history(conversation_id: UUID, session: Session) -> List[dict]:
    """Get recent conversation history"""
    messages = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(5)
    ).all()
    
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at
        }
        for msg in messages
    ]

@router.post("/", response_model=ConversationRead)
async def create_conversation(
    conversation: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    db_conversation = Conversation(
        title=conversation.title,
        created_by_id=current_user.id
    )
    session.add(db_conversation)
    session.commit()
    session.refresh(db_conversation)
    return db_conversation

@router.get("/", response_model=List[ConversationRead])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    conversations = session.exec(
        select(Conversation).where(Conversation.created_by_id == current_user.id)
    ).all()
    return conversations

@router.post("/messages/", response_model=MessageRead)
async def create_message(
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    logger.debug(f"Creating message for conversation {message.conversation_id}")
    try:
        # Verify conversation exists and user has access
        conversation = await verify_conversation_access(
            conversation_id=message.conversation_id,
            user_id=current_user.id,
            session=session
        )

        # Check if all documents are processed
        if not await check_documents_ready(conversation.id, session):
            raise APIError(
                status_code=status.HTTP_400_BAD_REQUEST,
                error="documents_processing",
                message="Please wait for document processing to complete"
            )

        # Create user message
        user_message = await create_user_message(
            content=message.content,
            conversation_id=message.conversation_id,
            user_id=current_user.id,
            session=session
        )

        # Create assistant message
        assistant_message = await create_assistant_message(
            conversation_id=message.conversation_id,
            user_id=current_user.id,
            session=session
        )

        # Generate response using RAG
        response_task = asyncio.create_task(
            generate_rag_response(
                message=user_message,
                conversation=conversation,
                assistant_message=assistant_message,
                session=session
            )
        )

        return user_message

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error creating message: {str(e)}")
        session.rollback()
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="message_creation_failed",
            message="Failed to create message",
            details={"error": str(e)}
        )

async def check_documents_ready(conversation_id: UUID, session: Session) -> bool:
    """Check if all documents in the conversation are processed and ready"""
    documents = session.exec(
        select(Document)
        .where(Document.conversation_id == conversation_id)
    ).all()
    
    return all(
        doc.status == DocumentStatus.COMPLETED and doc.is_searchable
        for doc in documents
    )

async def generate_rag_response(
    message: Message,
    conversation: Conversation,
    assistant_message: Message,
    session: Session
):
    try:
        # التحقق من إعدادات النموذج اللغوي قبل البدء
        settings = get_settings()
        logger.info(f"LLM CONFIG IN GENERATE_RESPONSE: Provider={settings.LLM_PROVIDER}, Model={settings.LLM_MODEL}")
        logger.info(f"LLM_BASE_URL={settings.LLM_BASE_URL}")
        
        # تحقق من أن النموذج المستخدم هو yehia-7b-preview-red فقط
        if settings.LLM_MODEL != "yehia-7b-preview-red":
            logger.error(f"WRONG MODEL CONFIGURATION: {settings.LLM_MODEL} instead of yehia-7b-preview-red")
            # قم بتصحيح الإعدادات إجبارياً
            settings.LLM_MODEL = "yehia-7b-preview-red"
            logger.info(f"Forced model to: {settings.LLM_MODEL}")
            
            # إعادة تهيئة LLM في خدمة RAG
            from core.llm.factory import ModelFactory
            rag_service.llm = ModelFactory.create_llm()
            rag_service.chat_model = ModelFactory.create_llm()
            logger.info("Re-initialized LLM with correct model")
        
        # تحقق من أن الرابط لا يشير إلى Hugging Face الافتراضي
        if "api-inference.huggingface.co" in settings.LLM_BASE_URL:
            logger.error(f"INVALID ENDPOINT URL: Using default HF API")
            # قم بتصحيح الرابط إجبارياً
            settings.LLM_BASE_URL = "https://ijt42iqbf30i3nly.us-east4.gcp.endpoints.huggingface.cloud/v1"
            logger.info(f"Forced endpoint URL to: {settings.LLM_BASE_URL}")
            
            # إعادة تهيئة LLM في خدمة RAG
            from core.llm.factory import ModelFactory
            rag_service.llm = ModelFactory.create_llm()
            rag_service.chat_model = ModelFactory.create_llm()
            logger.info("Re-initialized LLM with correct endpoint")
        
        # Initialize RAG service
        await rag_service.initialize()
        
        # Get conversation history for better context
        history = get_conversation_history(conversation.id, session)
        
        # Get relevant documents with improved hybrid search
        logger.info(f"Searching for documents relevant to: '{message.content}'")
        context_docs = await rag_service.hybrid_search(
            query=message.content,
            conversation_id=str(conversation.id),
            limit=5
        )
        
        if not context_docs:
            logger.warning(f"No relevant documents found for query: {message.content}")

        # التحقق من توفر النموذج اللغوي قبل استدعاء generate_response
        try:
            # إرسال استعلام بسيط للتحقق من أن النموذج يعمل
            test_response = await asyncio.wait_for(
                rag_service.llm.agenerate(["هل أنت متاح؟"], max_tokens=10),
                timeout=3.0
            )
            if not test_response.generations or not test_response.generations[0]:
                # إظهار خطأ صريح - LLM غير متاح
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="نموذج اللغة غير متاح حاليًا. يرجى المحاولة لاحقًا."
                )
                
            # التحقق من اسم النموذج المستخدم في الاستجابة
            model_info = getattr(test_response, 'llm_output', {}) or {}
            model_name = model_info.get('model_name', '') or getattr(rag_service.llm, 'model_name', '') or ''
            
            if model_name and 'mistral' in model_name.lower():
                logger.error(f"WRONG MODEL IN USE: {model_name}")
                # إعادة تهيئة النموذج مباشرة باستخدام OpenAI وتجاوز Factory
                from langchain_openai import ChatOpenAI
                from openai import OpenAI
                
                # استخدام OpenAI بشكل مباشر لتجاوز إعادة توجيه HuggingFace
                custom_client = OpenAI(
                    api_key=settings.HF_TOKEN or "hf_fake_token",
                    base_url=settings.LLM_BASE_URL
                )
                
                # إنشاء نموذج LangChain مع OpenAI
                rag_service.llm = ChatOpenAI(
                    model_name="tgi",  # اسم المستخدم من قبل النموذج المستضاف
                    openai_api_key=settings.HF_TOKEN or "hf_fake_token",
                    openai_api_base=settings.LLM_BASE_URL,
                    temperature=0.3,
                    max_tokens=800,
                    client=custom_client
                )
                rag_service.chat_model = rag_service.llm
                logger.info(f"Created direct OpenAI client for: {settings.LLM_BASE_URL}")
                
                # تحقق مجددًا بعد إعادة التهيئة
                test_response = await asyncio.wait_for(
                    rag_service.llm.agenerate(["تحقق سريع من نموذج yehia"], max_tokens=10),
                    timeout=3.0
                )
                logger.info(f"Reinitialized LLM test: {test_response.generations[0][0].text}")
            
        except Exception as e:
            logger.error(f"نموذج اللغة غير متاح: {e}")
            # إرجاع خطأ للعميل بدلاً من استجابة وهمية
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"نموذج اللغة غير متاح: {str(e)}"
            )
        
        # Generate response with full context
        response = await rag_service.generate_response(
            query=message.content,
            context_docs=context_docs,
            conversation_history=history
        )
        
        logger.info("Generated response successfully")

        # Update assistant message with response and source metadata
        assistant_message.content = response
        assistant_message.status = MessageStatus.COMPLETED
        
        # Include sources in message metadata for reference with enhanced details
        if context_docs:
            # Get source documents from the database to have complete information
            source_docs = []
            for doc in context_docs[:3]:  # Include top 3 sources
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
                
                source_docs.append({
                    "id": doc.get("id", ""),
                    "content": doc.get("content", ""),
                    "metadata": {
                        **metadata,
                        "file_name": file_name,  # Add explicit file name
                        "title": metadata.get("title", "Unknown Section") 
                    },
                    "score": doc.get("ensemble_score", 0) or 
                             doc.get("semantic_score", 0) or 
                             doc.get("cross_encoder_score", 0) or
                             doc.get("relevance_score", 0)
                })
            
            assistant_message.message_metadata = {
                "sources": source_docs
            }
            
        session.add(assistant_message)
        session.commit()
        session.refresh(assistant_message)
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        # Update message to show error
        assistant_message.content = "Sorry, I encountered an error while generating a response."
        assistant_message.status = MessageStatus.FAILED
        assistant_message.message_metadata = {"error": str(e)}
        session.add(assistant_message)
        session.commit()

@router.get("/{conversation_id}/messages/", response_model=List[MessageRead])
async def list_messages(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    # Check if conversation exists and belongs to user
    conversation = session.exec(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.created_by_id == current_user.id
        )
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found or you don't have permission to access it"
        )

    messages = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    ).all()
    
    return messages

@router.post("/{conversation_id}/attachments/", response_model=List[AttachmentRead])
async def upload_attachments(
    conversation_id: UUID,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    logger.debug(f"Uploading files to conversation {conversation_id}")
    
    try:
        # Verify conversation access
        conversation = await verify_conversation_access(
            conversation_id=conversation_id,
            user_id=current_user.id,
            session=session
        )

        uploaded_attachments = []
        
        for file in files:
            if not FileHandler.is_valid_file(file.filename, 0):
                continue
                
            try:
                # Save file
                file_path, file_size = await FileHandler.save_file(file, conversation_id)
                
                # Create attachment
                attachment = Attachment(
                    filename=file.filename,
                    file_path=file_path,
                    file_type=FileHandler.get_mime_type(file.filename),
                    file_size=file_size,
                    conversation_id=conversation_id,
                    uploaded_by_id=current_user.id
                )
                
                session.add(attachment)
                uploaded_attachments.append(attachment)
                
            except Exception as e:
                logger.error(f"Error uploading file {file.filename}: {str(e)}")
                continue
        
        if not uploaded_attachments:
            raise APIError(
                status_code=status.HTTP_400_BAD_REQUEST,
                error="no_files_uploaded",
                message="No valid files were uploaded"
            )
            
        # Commit to get attachment IDs
        session.commit()
        
        # Initialize RAG service
        await rag_service.initialize()
        
        # Process documents with progress feedback
        documents = await rag_service.process_attachments(
            attachments=uploaded_attachments,
            conversation_id=conversation_id,
            session=session
        )
        
        # Add system message about successful processing
        system_msg = f"✓ {len(documents)} document(s) processed and ready for questions."
        system_message = Message(
            content=system_msg,
            conversation_id=conversation_id,
            user_id=current_user.id,
            role=MessageRole.SYSTEM,
            status=MessageStatus.COMPLETED
        )
        session.add(system_message)
        session.commit()
        
        return uploaded_attachments
        
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        session.rollback()
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="upload_failed",
            message="Failed to process file upload",
            details={"error": str(e)}
        )

@router.get("/{conversation_id}/attachments/", response_model=List[AttachmentRead])
async def list_attachments(
    conversation_id: UUID,  # Changed from str to UUID
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    List all attachments for a specific conversation.
    """
    logger.debug(f"Fetching attachments for conversation {conversation_id}")
    
    try:
        # First verify conversation exists and user has access
        conversation = session.exec(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.created_by_id == current_user.id
            )
        ).first()
        
        if not conversation:
            logger.warning(f"User {current_user.id} attempted to access unauthorized conversation {conversation_id}")
            raise APIError(
                status_code=status.HTTP_404_NOT_FOUND,
                error="conversation_not_found",
                message="Conversation not found or you don't have permission to access it"
            )

        # Fetch attachments
        attachments = session.exec(
            select(Attachment)
            .where(Attachment.conversation_id == conversation_id)
            .order_by(Attachment.created_at)
        ).all()
        
        logger.info(f"Successfully retrieved {len(attachments)} attachments for conversation {conversation_id}")
        return attachments
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error fetching attachments: {str(e)}", exc_info=True)
        raise APIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="fetch_attachments_failed",
            message="Failed to fetch attachments",
            details={"error": str(e)}
        )

async def check_conversation_documents(conversation_id: UUID, session: Session) -> bool:
    """Check if conversation has any processed documents"""
    documents = session.exec(
        select(Document)
        .where(
            Document.conversation_id == conversation_id,
            Document.status == DocumentStatus.COMPLETED,
            Document.is_searchable
        )
    ).all()
    return len(documents) > 0

@router.post("/{conversation_id}/query")
async def process_query(
    conversation_id: str,
    query: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    try:
        logger.info(f"Processing query request: {query} for conversation: {conversation_id}")
        
        # سجل معلومات التكوين للتأكد من الإعدادات المستخدمة
        settings = get_settings()
        logger.info(f"LLM CONFIGURATION CHECK: Provider={settings.LLM_PROVIDER}, Model={settings.LLM_MODEL}")
        logger.info(f"LLM_BASE_URL={settings.LLM_BASE_URL}, HF_TOKEN available: {bool(settings.HF_TOKEN)}")
        
        # تحقق من أن النموذج المستخدم هو yehia-7b-preview-red فقط
        if settings.LLM_MODEL != "yehia-7b-preview-red":
            logger.error(f"WRONG MODEL CONFIGURATION: {settings.LLM_MODEL} instead of yehia-7b-preview-red")
            # قم بتصحيح الإعدادات إجبارياً
            settings.LLM_MODEL = "yehia-7b-preview-red"
            logger.info(f"Forced model to: {settings.LLM_MODEL}")
        
        # تحقق من أن الرابط لا يشير إلى Hugging Face الافتراضي
        if "api-inference.huggingface.co" in settings.LLM_BASE_URL:
            logger.error(f"INVALID ENDPOINT URL: Using default HF API")
            # قم بتصحيح الرابط إجبارياً
            settings.LLM_BASE_URL = "https://ijt42iqbf30i3nly.us-east4.gcp.endpoints.huggingface.cloud/v1"
            logger.info(f"Forced endpoint URL to: {settings.LLM_BASE_URL}")
            
        # Convert string to UUID
        try:
            conv_uuid = UUID(conversation_id.replace("-", ""))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={"error": "invalid_uuid", "message": "Invalid conversation ID format"}
            )
        
        # Verify conversation access and existence
        await verify_conversation_access(conv_uuid, current_user.id, session)
        
        # Check if conversation has processed documents
        has_documents = await check_conversation_documents(conv_uuid, session)
        if not has_documents:
            return {
                "response": "Please upload and process some documents before asking questions.",
                "context_docs": [],
                "conversation_id": conversation_id
            }
        
        # إعادة تهيئة LLM في خدمة RAG باستخدام النموذج الصحيح
        from core.llm.factory import ModelFactory
        # إنشاء نموذج جديد مع إعدادات صحيحة
        rag_service.llm = ModelFactory.create_llm()
        rag_service.chat_model = rag_service.llm
        logger.info("Re-initialized LLM with ModelFactory")
        
        # Initialize RAG service
        await rag_service.initialize()
        
        # Get relevant documents for the conversation
        context_docs = await rag_service.hybrid_search(
            query=query,
            conversation_id=str(conv_uuid),
            limit=5
        )
        
        # Get conversation history
        history = get_conversation_history(conv_uuid, session)
        
        # التحقق من توفر النموذج اللغوي قبل استدعاء generate_response
        try:
            # تسجيل معلومات تفصيلية عن النموذج الحالي
            logger.info(f"Current LLM instance: {rag_service.llm}")
            logger.info(f"LLM model_name: {getattr(rag_service.llm, 'model_name', 'unknown')}")
            
            # إرسال استعلام بسيط للتحقق من أن النموذج يعمل
            test_prompt = "هل أنت متاح؟"
            logger.info(f"Testing LLM with prompt: {test_prompt}")
            test_response = await asyncio.wait_for(
                rag_service.llm.agenerate([test_prompt], max_tokens=10),
                timeout=3.0
            )
            
            if not test_response.generations or not test_response.generations[0]:
                # إظهار خطأ صريح - LLM غير متاح
                logger.error("LLM test failed - no generations returned")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="نموذج اللغة غير متاح حاليًا. يرجى المحاولة لاحقًا."
                )
                
            logger.info(f"LLM test successful with response: {test_response.generations[0][0].text}")
        except Exception as e:
            logger.error(f"نموذج اللغة غير متاح: {e}")
            logger.error(traceback.format_exc())
            # إرجاع خطأ للعميل بدلاً من استجابة وهمية
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"نموذج اللغة غير متاح: {str(e)}"
            )
        
        # Generate response
        response = await rag_service.generate_response(
            query=query,
            context_docs=context_docs,
            conversation_history=history
        )
        
        return {
            "response": response,
            "context_docs": context_docs,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "query_processing_error", "message": str(e)}
        )

@router.post("/query")
async def chat_query(
    payload: Dict[str, Any] = Body(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """
    Enhanced chat query function that uses RAG services for better responses.
    """
    try:
        # Import manager at runtime to avoid circular imports
        from main import manager
        import os
        import traceback
        
        logger.info(f"Received chat query: {payload.get('query')}")
        
        query = payload.get("query", "")
        session_id = payload.get("session_id", "")
        history = payload.get("history", [])
        
        # سجل معلومات التكوين للتأكد من الإعدادات المستخدمة
        from core.config import get_settings
        settings = get_settings()
        logger.info(f"CHAT_QUERY LLM CONFIG: Provider={settings.LLM_PROVIDER}, Model={settings.LLM_MODEL}")
        logger.info(f"LLM_BASE_URL={settings.LLM_BASE_URL}, HF_TOKEN available: {bool(settings.HF_TOKEN)}")
        
        # تحقق من أن النموذج المستخدم هو yehia-7b-preview-red فقط
        if settings.LLM_MODEL != "yehia-7b-preview-red":
            logger.error(f"WRONG MODEL CONFIGURATION: {settings.LLM_MODEL} instead of yehia-7b-preview-red")
            # قم بتصحيح الإعدادات إجبارياً
            settings.LLM_MODEL = "yehia-7b-preview-red"
            logger.info(f"Forced model to: {settings.LLM_MODEL}")
            
        # تحقق من أن الرابط لا يشير إلى Hugging Face الافتراضي
        if "api-inference.huggingface.co" in settings.LLM_BASE_URL:
            logger.error(f"INVALID ENDPOINT URL DETECTED: {settings.LLM_BASE_URL}")
            # قم بتصحيح الرابط إجبارياً
            settings.LLM_BASE_URL = "https://ijt42iqbf30i3nly.us-east4.gcp.endpoints.huggingface.cloud/v1"
            logger.info(f"Forced endpoint URL to: {settings.LLM_BASE_URL}")
        
        # Update WebSocket with processing status
        if session_id:
            await manager.send_message({
                "task_state": "PROCESSING",
                "result": {
                    "status": "processing",
                    "message": "Processing query using RAG",
                    "state": {
                        "initialized": True,
                        "current_file": "",
                        "processing_status": {
                            "status": "PROCESSING",
                            "message": f"Processing query: {query[:30]}..."
                        },
                        "initialized_objects": {}
                    },
                    "session_id": session_id
                },
                "session_id": session_id,
                "pipeline_state": {
                    "initialized": True,
                    "processing_status": {
                        "status": "PROCESSING",
                        "message": "Processing query"
                    }
                }
            }, session_id)
        
        # Get documents associated with this conversation
        from apps.rag.models import Document
        
        try:
            # Handle different session_id formats (with/without hyphens)
            conv_id = session_id.replace("-", "")
            conversation_uuid = UUID(conv_id)
            
            documents = session.exec(
                select(Document).where(Document.conversation_id == conversation_uuid)
            ).all()
            
            logger.info(f"Found {len(documents)} documents for conversation {session_id}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            documents = []
        
        if not documents:
            response = "يرجى تحميل المستندات أولاً قبل طرح الأسئلة."
            
            # Return results
            result = {
                "fused_answer": response,
                "docs": [],
                "chat_history": [
                    *history,
                    {
                        "type": "ai",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    }
                ],
                "session_id": session_id
            }
            
            # Complete processing
            await manager.send_message({
                "task_state": "SUCCESS",
                "result": {
                    "status": "completed",
                    "message": "Query processed",
                    "state": {
                        "initialized": True,
                        "processing_status": {
                            "status": "COMPLETED",
                            "message": "Query processed"
                        }
                    },
                    "session_id": session_id
                },
                "session_id": session_id
            }, session_id)
            
            return result
        
        # إعادة تهيئة LLM في خدمة RAG باستخدام النموذج الصحيح
        from core.llm.factory import ModelFactory
        # إنشاء نموذج جديد مع إعدادات صحيحة
        rag_service.llm = ModelFactory.create_llm()
        rag_service.chat_model = rag_service.llm
        logger.info("Re-initialized LLM with ModelFactory")
        
        # Initialize RAG service
        await rag_service.initialize()
        
        # Get relevant documents for the conversation
        context_docs = await rag_service.hybrid_search(
            query=query,
            conversation_id=str(conversation_uuid),
            limit=5
        )
        
        # Get conversation history
        history = get_conversation_history(conversation_uuid, session)
        
        # التحقق من توفر النموذج اللغوي قبل استدعاء generate_response
        try:
            # تسجيل معلومات تفصيلية عن النموذج الحالي
            logger.info(f"Current LLM instance: {rag_service.llm}")
            logger.info(f"LLM model_name: {getattr(rag_service.llm, 'model_name', 'unknown')}")
            
            # إرسال استعلام بسيط للتحقق من أن النموذج يعمل
            test_prompt = "هل أنت متاح؟"
            logger.info(f"Testing LLM with prompt: {test_prompt}")
            test_response = await asyncio.wait_for(
                rag_service.llm.agenerate([test_prompt], max_tokens=10),
                timeout=3.0
            )
            
            if not test_response.generations or not test_response.generations[0]:
                # إظهار خطأ صريح - LLM غير متاح
                logger.error("LLM test failed - no generations returned")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="نموذج اللغة غير متاح حاليًا. يرجى المحاولة لاحقًا."
                )
                
            logger.info(f"LLM test successful with response: {test_response.generations[0][0].text}")
        except Exception as e:
            logger.error(f"نموذج اللغة غير متاح: {e}")
            logger.error(traceback.format_exc())
            # إرجاع خطأ للعميل بدلاً من استجابة وهمية
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"نموذج اللغة غير متاح: {str(e)}"
            )
        
        logger.info("Generating final response with context...")
        response = await rag_service.generate_response(
            query=query,
            context_docs=context_docs,
            conversation_history=history
        )
        
        # If still no proper response, do not create a basic one but show error
        if not response or response.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="فشل في توليد استجابة. يرجى المحاولة مرة أخرى."
            )
            
        logger.info(f"Generated response successfully with {len(response)} characters")
        
        # Add response to chat history
        new_history = [
            *history,
            {
                "type": "ai",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "context_docs": context_docs
            }
        ]
        
        # Create result object with the enhanced context documents
        result = {
            "fused_answer": response,
            "docs": context_docs,
            "chat_history": new_history,
            "session_id": session_id
        }
        
        # Update WebSocket with success state
        await manager.send_message({
            "task_state": "SUCCESS",
            "result": {
                "status": "completed",
                "message": "Query processed successfully",
                "state": {
                    "initialized": True,
                    "processing_status": {
                        "status": "COMPLETED",
                        "message": "Query processed successfully"
                    }
                },
                "session_id": session_id,
                "docs": context_docs,
                "fused_answer": response
            },
            "session_id": session_id
        }, session_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        
        # Send error message through WebSocket
        if session_id:
            try:
                await manager.send_message({
                    "task_state": "ERROR",
                    "result": {
                        "status": "error",
                        "message": f"Error processing query: {str(e)}",
                        "state": {
                            "initialized": True,
                            "processing_status": {
                                "status": "ERROR",
                                "message": "Query processing failed"
                            }
                        }
                    },
                    "session_id": session_id
                }, session_id)
            except Exception as ws_error:
                logger.error(f"Error sending WebSocket error message: {str(ws_error)}")
        
        # Return error to client
        return {
            "error": "query_processing_error",
            "message": str(e),
            "chat_history": history,
            "session_id": session_id
        }

@router.get("/recent-conversations", response_model=List[ConversationRead])
async def get_recent_conversations(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    limit: int = 10
):
    """
    Get the most recent conversations for the current user
    """
    # Query for conversations, ordered by update time
    conversations = session.exec(
        select(Conversation)
        .where(Conversation.created_by_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    ).all()
    
    # Enhance response with last message content
    result = []
    for conv in conversations:
        # Get the last message for each conversation
        last_message = session.exec(
            select(Message)
            .where(Message.conversation_id == conv.id)
            .order_by(Message.created_at.desc())
            .limit(1)
        ).first()
        
        # Create a dict from the conversation object
        conv_dict = {
            "id": conv.id,
            "title": conv.title,
            "created_by_id": conv.created_by_id,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "last_message": None
        }
        
        # Add last message if available
        if last_message:
            # Truncate message to a preview
            preview = last_message.content
            if len(preview) > 50:
                preview = preview[:50] + "..."
            
            # Add to dict
            conv_dict["last_message"] = preview
        
        # Create a ConversationRead object
        result.append(ConversationRead(**conv_dict))
    
    return result

@router.delete("/{conversation_id}", status_code=status.HTTP_200_OK)
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Delete a conversation and all its messages, attachments, and documents
    """
    try:
        # Verify conversation access
        conversation = await verify_conversation_access(
            conversation_id=conversation_id,
            user_id=current_user.id,
            session=session
        )
        
        logger.info(f"Deleting conversation {conversation_id} for user {current_user.id}")
        
        # 1. Delete all messages in the conversation
        messages = session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
        ).all()
        
        for message in messages:
            session.delete(message)
            
        # 2. Delete all attachments in the conversation
        attachments = session.exec(
            select(Attachment)
            .where(Attachment.conversation_id == conversation_id)
        ).all()
        
        for attachment in attachments:
            try:
                # Try to delete the file from disk
                file_path = attachment.file_path
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting attachment file: {e}")
            
            session.delete(attachment)
            
        # 3. Delete all documents associated with the conversation
        documents = session.exec(
            select(Document)
            .where(Document.conversation_id == conversation_id)
        ).all()
        
        for document in documents:
            try:
                # Use RAG service to properly delete document from vector store
                await rag_service.delete_document(document.id, session)
            except Exception as e:
                logger.error(f"Error deleting document from vector store: {e}")
                
                # Fallback: Just delete from database
                session.delete(document)
        
        # 4. Finally delete the conversation itself
        session.delete(conversation)
        
        # Commit all changes
        session.commit()
        
        return {
            "status": "success",
            "message": "Conversation deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}", exc_info=True)
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )

@router.delete("/clear-conversations", status_code=status.HTTP_200_OK)
async def clear_all_conversations(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Delete all conversations for the current user
    """
    try:
        logger.info(f"Clearing all conversations for user {current_user.id}")
        
        # Get all user's conversations
        conversations = session.exec(
            select(Conversation)
            .where(Conversation.created_by_id == current_user.id)
        ).all()
        
        if not conversations:
            return {
                "status": "success",
                "message": "No conversations to delete",
                "deleted_count": 0
            }
        
        total_conversations = len(conversations)
        deleted_count = 0
        
        # Delete each conversation
        for conversation in conversations:
            try:
                # 1. Delete all messages in the conversation
                messages = session.exec(
                    select(Message)
                    .where(Message.conversation_id == conversation.id)
                ).all()
                
                for message in messages:
                    session.delete(message)
                
                # 2. Delete all attachments in the conversation
                attachments = session.exec(
                    select(Attachment)
                    .where(Attachment.conversation_id == conversation.id)
                ).all()
                
                for attachment in attachments:
                    try:
                        # Try to delete the file from disk
                        file_path = attachment.file_path
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error deleting attachment file: {e}")
                    
                    session.delete(attachment)
                
                # 3. Find all documents associated with the conversation
                documents = session.exec(
                    select(Document)
                    .where(Document.conversation_id == conversation.id)
                ).all()
                
                for document in documents:
                    try:
                        # 3.1 First, find and delete all chunks related to this document
                        # هذه الخطوة ضرورية قبل حذف المستند بسبب قيد NOT NULL على حقل document_id
                        chunks = session.exec(
                            select(Chunk)
                            .where(Chunk.document_id == document.id)
                        ).all()
                        
                        for chunk in chunks:
                            session.delete(chunk)
                        
                        # Flush session to ensure chunk deletions are processed
                        session.flush()
                        
                        # 3.2 Then use RAG service to properly delete document from vector store
                        await rag_service.delete_document(document.id, session)
                    except Exception as e:
                        logger.error(f"Error deleting document from vector store: {e}", exc_info=True)
                        
                        try:
                            # Fallback: Just delete chunks and document from database directly
                            chunks = session.exec(
                                select(Chunk)
                                .where(Chunk.document_id == document.id)
                            ).all()
                            
                            for chunk in chunks:
                                session.delete(chunk)
                            
                            # Delete the document after chunks
                            session.delete(document)
                        except Exception as inner_e:
                            logger.error(f"Error during fallback document deletion: {inner_e}", exc_info=True)
                
                # 4. Finally delete the conversation itself
                session.delete(conversation)
                deleted_count += 1
                
                # Commit after each conversation to avoid large transaction
                session.commit()
                
            except Exception as e:
                logger.error(f"Error deleting conversation {conversation.id}: {e}", exc_info=True)
                # Rollback the current conversation's transaction
                session.rollback()
                # Continue with next conversation instead of failing completely
                continue
        
        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_count} conversations out of {total_conversations}",
            "deleted_count": deleted_count,
            "total_count": total_conversations
        }
        
    except Exception as e:
        logger.error(f"Error clearing conversations: {e}", exc_info=True)
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversations: {str(e)}"
        )