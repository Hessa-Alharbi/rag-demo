from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Body, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from core.db import init_db
from core.logger import logger
from core.config import get_settings  # Changed from core.settings to core.config
from core.middleware import error_handler
from core.vector_store.connection import MilvusConnectionManager
from core.startup import initialize_vector_stores
from typing import Dict, List, Any
import json
import os
from uuid import uuid4
import shutil
from apps.auth.routes import get_current_user
from core.db import get_session
from sqlmodel import Session
from apps.chat.schemas import ConversationRead

from apps.admin.routes import router as admin_router
from apps.auth.routes import router as auth_router
from apps.chat.routes import router as chat_router
from apps.organizations.routes import router as org_router
from apps.test.routes import router as test_router

# إضافة router لتطبيق RAG
from apps.rag.routes import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    # Validate settings on startup
    try:
        settings = get_settings()
        settings.validate_settings()
        # Initialize the ML model
        init_db()
        print("Loading ML model")
        
        # Initialize Milvus connection
        MilvusConnectionManager.ensure_connection()
        
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Close the ML model
        print("Closing ML model")
        
        # Cleanup
        MilvusConnectionManager.close_connection()
        logger.info("Application shutdown complete")


app = FastAPI(lifespan=lifespan, title="Navid RAG API", root_path="/api")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://f3c5-2a02-cb80-4271-93aa-dc2c-9eea-2d8e-7325.ngrok-free.app", "http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await initialize_vector_stores()

# Add middleware
app.middleware("http")(error_handler)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(chat_router, prefix="/conversations", tags=["conversations"])
app.include_router(org_router, prefix="/organizations", tags=["organizations"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])
app.include_router(test_router, prefix="/test", tags=["test"])

# إضافة router لتطبيق RAG
app.include_router(rag_router, prefix="/rag", tags=["rag"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def send_message(self, message: Dict[str, Any], task_id: str):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                await connection.send_text(json.dumps(message))

manager = ConnectionManager()

@app.websocket("/ws/task/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await manager.connect(websocket, task_id)
    try:
        # Check if this is a session ID (from a conversation)
        if task_id.startswith("chat_") or "-" in task_id:
            # For session IDs, just use the same ID as both task_id and session_id
            session_id = task_id
        else:
            # For task IDs, we'll set session_id later when we process the upload
            session_id = ""
            
        # On connection, send initial state
        await manager.send_message({
            "task_state": "PROCESSING",
            "result": {
                "status": "processing",
                "message": "Task is being processed",
                "state": {
                    "initialized": True,  # Change to TRUE to avoid frontend error
                    "current_file": "",
                    "processing_status": {
                        "status": "INITIALIZING",
                        "message": "Initializing task processing"
                    },
                    "initialized_objects": {}
                },
                "session_id": session_id
            },
            "session_id": session_id,
            "pipeline_state": {
                "initialized": True,  # Change to TRUE to avoid frontend error
                "current_file": "",
                "processing_status": {
                    "status": "INITIALIZING",
                    "message": "Initializing pipeline"
                },
                "initialized_objects": {}
            }
        }, task_id)
        
        # Wait for messages (not used in this simple implementation)
        while True:
            data = await websocket.receive_text()
            # Process message if needed
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, task_id)

# Add a direct file upload endpoint for frontend integration
@app.post("/initialize")
async def initialize_chat(file: UploadFile = File(...), session: Session = Depends(get_session), current_user = Depends(get_current_user)):
    try:
        # Generate a task ID and a session ID
        task_id = f"task_{uuid4()}"
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join("uploads", task_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File saved at {file_path}")
        
        # Create a conversation in the database with the current user ID
        from apps.chat.models import Conversation
        conversation = Conversation(
            id=uuid4(),
            title=f"Chat about {file.filename}",
            created_by_id=current_user.id  # Make sure current_user.id is not None
        )
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        
        conversation_id = str(conversation.id)
        logger.info(f"Created conversation with ID: {conversation_id}")
        
        # Create a simplified document record without vector embeddings
        from apps.rag.models import Document, DocumentStatus
        
        # Create document directly without vector store processing
        doc = Document(
            id=uuid4(),
            conversation_id=conversation.id,
            title=os.path.basename(file_path),
            source=file_path,
            type=os.path.splitext(file.filename)[1].lower().replace('.', ''),
            status=DocumentStatus.COMPLETED,  # Mark as completed 
            is_searchable=True,  # Mark as searchable
            document_metadata={
                "filename": file.filename,
                "type": "uploaded_document",
                "simplified_processing": True,
            },
            created_by_id=current_user.id
        )
        
        # Extract text content (simplified)
        try:
            if file.filename.lower().endswith('.pdf'):
                from langchain_community.document_loaders import PyPDFLoader
                # استخدام PyPDFLoader مع الحفاظ على النص الأصلي دون تعديل
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                text = "\n\n".join(page.page_content for page in pages)
                # نحتفظ بالنص كما هو دون أي معالجة إضافية
                doc.content = text
            elif file.filename.lower().endswith(('.docx', '.doc')):
                import docx2txt
                doc.content = docx2txt.process(file_path)
            elif file.filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    doc.content = txt_file.read()
            else:
                doc.content = f"[File content from {file.filename}]"
        except Exception as extract_error:
            logger.error(f"Error extracting text: {str(extract_error)}")
            doc.content = f"[Error extracting content from {file.filename}]"
        
        # Save the document
        session.add(doc)
        session.commit()
        
        # إضافة فهرسة المستند مباشرة باستخدام RAG service بدلاً من الاعتماد على عملية منفصلة
        try:
            # تهيئة خدمة RAG
            from apps.rag.services import RAGService
            rag_service = RAGService()
            await rag_service.initialize()
            
            # الحصول على مخزن المتجهات للمحادثة
            from core.vector_store.singleton import VectorStoreSingleton
            vector_store = await VectorStoreSingleton.get_conversation_store(str(conversation.id))
            
            # تقسيم المستند إلى أجزاء
            content = doc.content
            chunks = rag_service.semantic_chunker.split_text(content)
            chunk_titles = rag_service.semantic_chunker.generate_chunk_titles(chunks)
            
            # تحضير الأجزاء لقاعدة المتجهات
            texts = []
            metadatas = []
            
            for i, (chunk_text, chunk_title) in enumerate(zip(chunks, chunk_titles)):
                # تحديد إذا كان النص عربي
                from core.language.arabic_utils import ArabicTextProcessor
                is_arabic = ArabicTextProcessor.contains_arabic(chunk_text)
                
                # إضافة النص الأصلي
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
                
                # إذا كان النص عربي، أضف نسخًا معالجة للبحث الأفضل
                if is_arabic:
                    # نضيف نسخة إضافية مع إزالة التشكيل والتطويل فقط
                    # ولكن دون تغيير أي حرف
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
                    
                    # نحذف نسخة تبديل الحروف الشائعة لأنها تغير الحروف الأصلية
                    
                    # استخرج وفهرس الكلمات المفتاحية
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
            
            # إضافة إلى مخزن المتجهات
            if texts:
                vector_ids = await vector_store.add_documents(texts, metadatas)
                
                # تحديث المستند
                doc.vector_ids = vector_ids
                session.add(doc)
                session.commit()
                
                logger.info(f"تمت فهرسة المستند {doc.id} بنجاح مع {len(vector_ids)} متجهات")
        
        except Exception as e:
            logger.error(f"خطأ أثناء فهرسة المستند: {e}")
            # المستند تم حفظه بالفعل، فنستمر رغم فشل الفهرسة
        
        # Update websocket with success status immediately
        await manager.send_message({
            "task_state": "SUCCESS",
            "result": {
                "status": "completed",
                "message": "File processed successfully",
                "state": {
                    "initialized": True,
                    "current_file": file.filename,
                    "processing_status": {
                        "status": "COMPLETED",
                        "message": "Document saved successfully"
                    },
                    "initialized_objects": {}
                },
                "session_id": conversation_id
            },
            "session_id": conversation_id,
            "pipeline_state": {
                "initialized": True,
                "processing_status": {
                    "status": "COMPLETED",
                    "message": "Pipeline completed"
                }
            }
        }, task_id)
            
        return {
            "message": "File uploaded successfully. Processing completed.",
            "task_id": task_id,
            "file_path": file_path,
            "session_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to initialize chat: {str(e)}"}
        )

# Add a direct query endpoint for frontend integration
@app.post("/query")
async def handle_query(
    payload: Dict[str, Any] = Body(...),
    session: Session = Depends(get_session),
    current_user = Depends(get_current_user)
):
    """
    Endpoint to handle chat queries directly from the frontend
    This will call the actual chat_query function in apps.chat.routes
    """
    from apps.chat.routes import chat_query
    
    # Pass the request to the chat_query function
    return await chat_query(payload, session, current_user)

# Add a direct endpoint for recent conversations
@app.get("/recent-conversations", response_model=List[ConversationRead])
async def get_recent_conversations_route(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session),
    limit: int = 10
):
    """
    Top-level route for getting recent conversations that forwards to the chat router handler
    """
    from apps.chat.routes import get_recent_conversations
    return await get_recent_conversations(current_user, session, limit)

# Add a direct endpoint for clearing all conversations
@app.delete("/clear-conversations", status_code=status.HTTP_200_OK)
async def clear_all_conversations_route(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Top-level route for clearing all conversations that forwards to the chat router handler
    """
    from apps.chat.routes import clear_all_conversations
    return await clear_all_conversations(current_user, session)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)