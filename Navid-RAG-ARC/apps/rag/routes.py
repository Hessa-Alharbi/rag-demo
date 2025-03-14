from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from loguru import logger

from apps.auth.models import User
from apps.auth.routes import get_current_user
from core.db import get_session
from .models import Document
from .services import RAGService

router = APIRouter(prefix="/documents", tags=["Documents"])
rag_service = RAGService()


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Delete a document and clean up associated vector store entries
    """
    try:
        # Check if document exists and user has access to it
        document_query = select(Document).where(
            Document.id == document_id, 
            Document.created_by_id == current_user.id
        )
        document = session.exec(document_query).first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or you don't have permission to delete it"
            )
        
        # Initialize RAG service
        await rag_service.initialize()
        
        # Delete document and associated vector store entries
        success = await rag_service.delete_document(document_id, session)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete document"
            )
        
        return {"status": "success", "message": "Document successfully deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.delete("/conversation/{conversation_id}", status_code=status.HTTP_200_OK)
async def delete_conversation_documents(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Delete all documents associated with a conversation
    """
    try:
        # Check if conversation has documents and user has access to them
        documents_query = select(Document).where(
            Document.conversation_id == conversation_id,
            Document.created_by_id == current_user.id
        )
        documents = session.exec(documents_query).all()
        
        if not documents:
            return {"status": "success", "message": "No documents found for this conversation"}
        
        # Initialize RAG service
        await rag_service.initialize()
        
        # Delete each document
        deleted_count = 0
        for document in documents:
            success = await rag_service.delete_document(document.id, session)
            if success:
                deleted_count += 1
                
        return {
            "status": "success", 
            "message": f"Successfully deleted {deleted_count} documents from conversation",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error deleting conversation documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation documents: {str(e)}"
        )