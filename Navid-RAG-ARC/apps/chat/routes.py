from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from typing import List
from uuid import UUID
from sqlmodel import Session, select
from apps.chat.models import Conversation, Message, Attachment, MessageRole, MessageStatus
from apps.chat.schemas import ConversationCreate, ConversationRead, MessageCreate, MessageRead, AttachmentRead
from apps.chat.utils import FileHandler
from apps.users.models import User
from core.db import get_session
from apps.auth.routes import get_current_user
from core.logger import logger
from core.errors import APIError
from apps.rag.services import RAGService
import asyncio
from apps.rag.models import Document, DocumentStatus


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
        
        # Include sources in message metadata for reference
        if context_docs:
            assistant_message.message_metadata = {
                "sources": [
                    {
                        "id": doc.get("id", ""),
                        "metadata": doc.get("metadata", {})
                    }
                    for doc in context_docs[:3]  # Include top 3 sources
                ]
            }
            
        session.add(assistant_message)
        session.commit()
        session.refresh(assistant_message)

    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        assistant_message.status = MessageStatus.FAILED
        assistant_message.message_metadata = {"error": str(e)}
        session.add(assistant_message)
        session.commit()
        session.refresh(assistant_message)
        raise

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
            Document.is_searchable == True
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
        
        # Get relevant documents for the conversation
        context_docs = await rag_service.hybrid_search(
            query=query,
            conversation_id=conversation_id,
            limit=5
        )
        
        # Get conversation history
        history = get_conversation_history(conv_uuid, session)
        
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