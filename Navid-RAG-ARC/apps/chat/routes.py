from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Body
from typing import List, Dict, Any, Optional
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
from apps.rag.models import Document, DocumentStatus, Chunk
from datetime import datetime
import sys
import os


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
        
        logger.info(f"Received chat query: {payload.get('query')}")
        
        query = payload.get("query", "")
        session_id = payload.get("session_id", "")
        history = payload.get("history", [])
        
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
        
        # Use RAG service for better search and response generation
        await rag_service.initialize()
        
        # Get conversation history for context
        conv_history = []
        if history:
            for msg in history:
                conv_history.append({
                    "role": "user" if msg.get("type") == "human" else "assistant",
                    "content": msg.get("content", ""),
                    "created_at": msg.get("timestamp", datetime.now().isoformat())
                })
        
        # Use hybrid search to find relevant document sections
        logger.info(f"Performing hybrid search for query: '{query}'")
        context_docs = await rag_service.hybrid_search(
            query=query,
            conversation_id=str(conversation_uuid),
            limit=5
        )
        
        # Prepare context docs with enhanced metadata
        enhanced_context_docs = []
        for doc in context_docs:
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
                
            enhanced_context_docs.append({
                "content": doc.get("content", ""),
                "metadata": {
                    **metadata,
                    "file_name": file_name,
                    "title": metadata.get("title", "Unknown Section")
                },
                "score": doc.get("ensemble_score", 0) or 
                        doc.get("semantic_score", 0) or 
                        doc.get("cross_encoder_score", 0) or
                        doc.get("relevance_score", 0)
            })
        
        if not enhanced_context_docs:
            logger.warning(f"No relevant documents found for query: {query}")
            enhanced_context_docs = []
            
            # Check if query is in Arabic
            is_arabic = any(c for c in query if '\u0600' <= c <= '\u06FF')
            
            # Fallback to improved text search if hybrid search returns nothing
            for doc in documents:
                if not doc.content:
                    continue
                
                # Use enhanced text search approach
                if is_arabic:
                    # Import ArabicTextProcessor for better Arabic handling
                    from core.language.arabic_utils import ArabicTextProcessor
                    
                    # Normalize query for better matching
                    normalized_query = ArabicTextProcessor.normalize_arabic(query)
                    query_keywords = ArabicTextProcessor.preprocess_arabic_query(query)["keywords"]
                    
                    # Normalize document content
                    normalized_content = ArabicTextProcessor.normalize_arabic(doc.content)
                    paragraphs = normalized_content.split('\n\n')
                    
                    for i, paragraph in enumerate(paragraphs):
                        # Match on normalized text and keywords for better recall
                        if normalized_query in paragraph or any(keyword in paragraph for keyword in query_keywords if keyword):
                            # Add this paragraph as a "document" with metadata
                            original_paragraph = doc.content.split('\n\n')[i] if i < len(doc.content.split('\n\n')) else paragraph
                            
                            # Get file name from metadata or title
                            file_name = doc.title
                            if hasattr(doc, 'doc_metadata') and doc.doc_metadata:
                                if isinstance(doc.doc_metadata, dict):
                                    file_name = doc.doc_metadata.get('original_filename', doc.title)
                            
                            enhanced_context_docs.append({
                                "content": original_paragraph,
                                "metadata": {
                                    "document_id": str(doc.id),
                                    "title": doc.title,
                                    "paragraph": i,
                                    "file_name": file_name
                                },
                                "score": 2.0 if normalized_query in paragraph else 1.0  # Higher score for better matches
                            })
                            
                            # Limit to 5 paragraphs per document
                            if len(enhanced_context_docs) >= 5:
                                break
                else:
                    # Simple text search for non-Arabic - find paragraphs that contain any of the query words
                    query_words = query.lower().split()
                    paragraphs = doc.content.split('\n\n')
                    
                    for i, paragraph in enumerate(paragraphs):
                        if any(word.lower() in paragraph.lower() for word in query_words):
                            # Get file name from metadata or title
                            file_name = doc.title
                            if hasattr(doc, 'doc_metadata') and doc.doc_metadata:
                                if isinstance(doc.doc_metadata, dict):
                                    file_name = doc.doc_metadata.get('original_filename', doc.title)
                            
                            # Add this paragraph as a "document" with metadata
                            enhanced_context_docs.append({
                                "content": paragraph,
                                "metadata": {
                                    "document_id": str(doc.id),
                                    "title": doc.title,
                                    "paragraph": i,
                                    "file_name": file_name
                                },
                                "score": 1.0  # Dummy score
                            })
                            
                            # Limit to 5 paragraphs per document
                            if len(enhanced_context_docs) >= 5:
                                break
        
        # Generate response using RAG
        logger.info("Generating response with RAG service")
        response = await rag_service.generate_response(
            query=query,
            context_docs=enhanced_context_docs,
            conversation_history=conv_history
        )
        
        # If still no proper response, create a basic one
        if not response or response.strip() == "":
            if enhanced_context_docs:
                # Create context by joining paragraphs
                context = "\n\n".join([d.get("content", "") for d in enhanced_context_docs])
                response = f"وجدت المعلومات التالية حول '{query}':\n\n{context[:1000]}..."
            else:
                response = "لم أتمكن من العثور على معلومات كافية حول استفسارك."
        
        # Add response to chat history
        new_history = [
            *history,
            {
                "type": "ai",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "context_docs": enhanced_context_docs
            }
        ]
        
        # Create result object with the enhanced context documents
        result = {
            "fused_answer": response,
            "docs": enhanced_context_docs,
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
                "docs": enhanced_context_docs,
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