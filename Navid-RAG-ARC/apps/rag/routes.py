from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from loguru import logger

from apps.users.models import User
from apps.auth.routes import get_current_user
from core.db import get_session
from .models import Document
from .services import RAGService
from apps.chat.models import Conversation

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

@router.post("/reindex/{conversation_id}")
async def reindex_conversation_documents(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    يقوم بإعادة فهرسة جميع المستندات في المحادثة.
    يستخدم عندما تكون المستندات موجودة في قاعدة البيانات لكن غير مفهرسة في vector store.
    """
    try:
        # تحقق من وجود المحادثة وملكية المستخدم
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

        # احصل على جميع المستندات
        documents = session.exec(
            select(Document)
            .where(Document.conversation_id == conversation_id)
        ).all()
        
        if not documents:
            return {"message": "No documents found for this conversation", "reindexed": 0}
            
        # تهيئة خدمة RAG
        await rag_service.initialize()
        
        reindexed_count = 0
        errors = []
        
        # إعادة فهرسة كل مستند
        for doc in documents:
            try:
                # تحويل المستند إلى متجهات
                if len(doc.vector_ids) == 0:
                    # استخدم عرض content الموجود مسبقاً
                    content = doc.content
                    
                    # قم بتقسيم المحتوى إلى أجزاء
                    chunks = rag_service.semantic_chunker.split_text(content)
                    
                    # احصل على العناوين للأجزاء
                    chunk_titles = rag_service.semantic_chunker.generate_chunk_titles(chunks)
                    
                    # تحضير الأجزاء لقاعدة المتجهات
                    texts = []
                    metadatas = []
                    db_chunks = []
                    
                    for i, (chunk_text, chunk_title) in enumerate(zip(chunks, chunk_titles)):
                        # تحقق إذا كان النص عربي
                        from core.language.arabic_utils import ArabicTextProcessor
                        is_arabic = ArabicTextProcessor.contains_arabic(chunk_text)
                        
                        # إضافة النص الأصلي أولاً
                        texts.append(chunk_text)
                        metadatas.append({
                            "document_id": str(doc.id),
                            "conversation_id": str(doc.conversation_id),
                            "chunk_index": i,
                            "title": chunk_title,
                            "doc_title": doc.title,
                            "is_arabic": is_arabic,
                            "variant": "original"
                        })
                        
                        # إذا كان النص عربي، أضف نسخاً معالجة للبحث الأفضل
                        if is_arabic:
                            # أضف النسخة المطبعة
                            normalized_text = ArabicTextProcessor.normalize_arabic(chunk_text)
                            texts.append(normalized_text)
                            metadatas.append({
                                "document_id": str(doc.id),
                                "conversation_id": str(doc.conversation_id),
                                "chunk_index": i,
                                "title": chunk_title,
                                "doc_title": doc.title,
                                "is_arabic": is_arabic,
                                "variant": "normalized"
                            })
                            
                            # أضف نسخة مع تبديل الحروف الشائعة
                            variant_text = normalized_text.replace("ة", "ه").replace("ى", "ي").replace("أ", "ا").replace("إ", "ا")
                            texts.append(variant_text)
                            metadatas.append({
                                "document_id": str(doc.id),
                                "conversation_id": str(doc.conversation_id),
                                "chunk_index": i,
                                "title": chunk_title,
                                "doc_title": doc.title,
                                "is_arabic": is_arabic,
                                "variant": "variant"
                            })
                            
                            # استخرج وفهرس العبارات الرئيسية بشكل منفصل للمطابقة الأفضل
                            key_phrases = ArabicTextProcessor.extract_arabic_keywords(chunk_text, max_keywords=5)
                            if key_phrases:
                                key_phrase_text = " ".join(key_phrases)
                                texts.append(key_phrase_text)
                                metadatas.append({
                                    "document_id": str(doc.id),
                                    "conversation_id": str(doc.conversation_id),
                                    "chunk_index": i,
                                    "title": chunk_title,
                                    "doc_title": doc.title,
                                    "is_arabic": is_arabic,
                                    "variant": "keywords"
                                })
                    
                    # الحصول على مخزن المتجهات الخاص بالمحادثة
                    from core.vector_store.singleton import VectorStoreSingleton
                    vector_store = await VectorStoreSingleton.get_conversation_store(str(doc.conversation_id))
                    
                    # إضافة إلى مخزن المتجهات
                    vector_ids = await vector_store.add_documents(texts, metadatas)
                    
                    # تحديث المستند
                    doc.vector_ids = vector_ids
                    doc.is_searchable = True
                    session.add(doc)
                    reindexed_count += 1
                    
                    logger.info(f"Successfully reindexed document {doc.id} with {len(vector_ids)} vectors")
            except Exception as e:
                logger.error(f"Error reindexing document {doc.id}: {e}")
                errors.append({"document_id": str(doc.id), "error": str(e)})
        
        session.commit()
        
        return {
            "message": f"Reindexed {reindexed_count} documents",
            "reindexed": reindexed_count,
            "total": len(documents),
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error reindexing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reindexing documents: {str(e)}"
        )

@router.get("/check_vectors/{document_id}")
async def check_document_vectors(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    التحقق من وجود المتجهات وعددها للمستند المحدد
    """
    try:
        # الحصول على المستند
        document = session.exec(
            select(Document).where(Document.id == document_id)
        ).first()
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {document_id} not found"
            )
            
        # التحقق من ملكية المستخدم للمستند
        conversation = session.exec(
            select(Conversation)
            .where(
                Conversation.id == document.conversation_id,
                Conversation.created_by_id == current_user.id
            )
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to access this document"
            )
            
        # إرجاع معلومات حول المتجهات وحالة المستند
        return {
            "document_id": str(document.id),
            "title": document.title,
            "has_vectors": len(document.vector_ids) > 0,
            "vector_count": len(document.vector_ids),
            "is_searchable": document.is_searchable,
            "status": document.status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking document vectors: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error checking document vectors: {str(e)}"
        )