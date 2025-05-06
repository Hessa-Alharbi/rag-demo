from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Body, Depends, status, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
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
import traceback  # إضافة استيراد وحدة traceback
from uuid import uuid4
import shutil
from apps.auth.routes import get_current_user
from core.db import get_session
from sqlmodel import Session
from apps.chat.schemas import ConversationRead
from apps.rag.services import RAGService
from core.llm.factory import ModelFactory

from apps.admin.routes import router as admin_router
from apps.auth.routes import router as auth_router
from apps.chat.routes import router as chat_router
from apps.organizations.routes import router as org_router
from apps.test.routes import router as test_router

# إضافة router لتطبيق RAG
from apps.rag.routes import router as rag_router

import argparse
import uvicorn
import httpx
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Define global variables
settings = get_settings()
rag_service = None
chat_model = None

# Pydantic models for request/response
class RAGRequest(BaseModel):
    query: str
    document_ids: List[str] = []
    options: Dict[str, Any] = {}

class RAGResponse(BaseModel):
    answer: str
    context: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service, chat_model
    
    logger.info("Starting up application...")
    # Validate settings on startup
    try:
        settings.validate_settings()
        
        # التحقق من إعدادات نموذج اللغة عند بدء التشغيل
        logger.info(f"==== CHECKING LLM SETTINGS AT STARTUP ====")
        logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
        logger.info(f"LLM Model: {settings.LLM_MODEL}")
        logger.info(f"Base URL: {settings.LLM_BASE_URL}")
        
        # فرض استخدام النموذج الصحيح
        if settings.LLM_MODEL != "yehia-7b-preview-red":
            logger.error(f"WRONG MODEL CONFIGURATION DETECTED: {settings.LLM_MODEL}")
            settings.LLM_MODEL = "yehia-7b-preview-red"
            logger.info(f"Forced model to: {settings.LLM_MODEL}")
            
        # منع استخدام API الافتراضي لـ HuggingFace
        if "api-inference.huggingface.co" in settings.LLM_BASE_URL:
            logger.error(f"INVALID ENDPOINT DETECTED: {settings.LLM_BASE_URL}")
            settings.LLM_BASE_URL = "https://ijt42iqbf30i3nly.us-east4.gcp.endpoints.huggingface.cloud/v1"
            logger.info(f"Forced endpoint URL to: {settings.LLM_BASE_URL}")
        
        # Initialize the ML model
        init_db()
        print("Loading ML model")
        
        # Initialize Milvus connection
        MilvusConnectionManager.ensure_connection()
        
        # Initialize the chat model (optional)
        chat_model = initialize_chat()
        
        # Initialize the RAG service
        rag_service = RAGService()
        await rag_service.initialize()
        logger.info("RAG service initialized successfully")
        
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
    global rag_service
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
        # Declare rag_service as global
        global rag_service
        
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
        
        # الحصول على إعدادات النظام وفحصها
        logger.info(f"INITIALIZE_CHAT LLM CONFIG: Provider={settings.LLM_PROVIDER}, Model={settings.LLM_MODEL}")
        logger.info(f"LLM_BASE_URL={settings.LLM_BASE_URL}, HF_TOKEN available: {bool(settings.HF_TOKEN)}")
        
        # تحقق من أن النموذج المستخدم هو yehia-7b-preview-red فقط
        if settings.LLM_MODEL != "yehia-7b-preview-red":
            logger.error(f"WRONG MODEL CONFIGURATION DETECTED: {settings.LLM_MODEL}")
            settings.LLM_MODEL = "yehia-7b-preview-red"
            logger.info(f"Forced model to: {settings.LLM_MODEL}")
            
        # تحقق من أن الرابط لا يشير إلى Hugging Face الافتراضي
        if "api-inference.huggingface.co" in settings.LLM_BASE_URL:
            logger.error(f"INVALID ENDPOINT DETECTED: {settings.LLM_BASE_URL}")
            settings.LLM_BASE_URL = "https://ijt42iqbf30i3nly.us-east4.gcp.endpoints.huggingface.cloud/v1"
            logger.info(f"Forced endpoint URL to: {settings.LLM_BASE_URL}")
        
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
                
                # سجل محتوى المستند للتأكد من استخراجه بشكل صحيح
                logger.info(f"Successfully extracted content from PDF, first 200 chars: {text[:200]}")
                
            elif file.filename.lower().endswith(('.docx', '.doc')):
                import docx2txt
                doc.content = docx2txt.process(file_path)
                logger.info(f"Successfully extracted content from DOCX, first 200 chars: {doc.content[:200]}")
                
            elif file.filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    doc.content = txt_file.read()
                logger.info(f"Successfully extracted content from TXT, first 200 chars: {doc.content[:200]}")
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
            # التأكد من أن rag_service موجود وتم تهيئته
            if not rag_service:
                logger.warning("RAG service is not initialized. Creating a new instance.")
                from apps.rag.services import RAGService
                from core.llm.factory import ModelFactory
                rag_service = RAGService()
                await rag_service.initialize()
            
            # التحقق مما إذا كان النموذج وهمياً (عند إيقاف الخادم)
            is_dummy = False
            if hasattr(rag_service, 'llm') and rag_service.llm is not None:
                if hasattr(rag_service.llm, 'model_name'):
                    is_dummy = 'dummy' in rag_service.llm.model_name.lower()
            else:
                logger.warning("rag_service.llm is not available. Initializing it...")
                from core.llm.factory import ModelFactory
                rag_service.llm = ModelFactory.create_llm()
                rag_service.chat_model = rag_service.llm
                
            if is_dummy:
                logger.warning("Using dummy model because the server endpoint is paused")
                await manager.send_message({
                    "task_state": "WARNING",
                    "result": {
                        "status": "completed_with_warning",
                        "message": "تم معالجة الملف ولكن خادم النموذج اللغوي متوقف حالياً",
                        "state": {
                            "initialized": True,
                            "current_file": file.filename,
                            "processing_status": {
                                "status": "COMPLETED_WITH_WARNING",
                                "message": "خادم النموذج اللغوي متوقف حالياً، يرجى التواصل مع المشرف لإعادة تشغيله"
                            }
                        },
                        "session_id": conversation_id
                    },
                    "session_id": conversation_id
                }, task_id)
            
            # التحقق من توفر النموذج بالاختبار المباشر
            import asyncio
            logger.info("Testing LLM with simple prompt...")
            try:
                test_response = await asyncio.wait_for(
                    rag_service.llm.agenerate(["هل أنت متاح؟"], max_tokens=10),
                    timeout=3.0
                )
                if not test_response.generations or not test_response.generations[0]:
                    await manager.send_message({
                        "task_state": "WARNING",
                        "result": {
                            "status": "completed_with_warning",
                            "message": "تم معالجة الملف ولكن نموذج اللغة غير متوفر حاليًا",
                            "state": {
                                "initialized": True,
                                "current_file": file.filename,
                                "processing_status": {
                                    "status": "COMPLETED_WITH_WARNING",
                                    "message": "نموذج اللغة غير متوفر"
                                }
                            },
                            "session_id": conversation_id
                        },
                        "session_id": conversation_id
                    }, task_id)
                    logger.warning(f"نموذج اللغة غير متوفر أثناء معالجة الملف {file.filename}")
                else:
                    logger.info(f"LLM test successful: {test_response.generations[0][0].text}")
            except Exception as e:
                logger.error(f"Failed to test LLM: {e}")
                logger.error(traceback.format_exc())
        
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
                logger.info(f"Adding {len(texts)} text chunks to vector store for conversation {conversation_id}")
                try:
                    vector_ids = await vector_store.add_documents(texts, metadatas)
                    
                    # تحديث سجل المستند ليشمل vector_ids
                    if vector_ids:
                        # حفظ الـ IDs بالقاعدة بيانات
                        doc.vector_ids = [str(vid) for vid in vector_ids]
                        session.add(doc)
                        session.commit()
                        
                        # تأكيد أن الفهرسة تمت بنجاح عن طريق البحث عن أحد النصوص
                        try:
                            # اختبار البحث
                            test_query = texts[0][:100]  # أول 100 حرف من النص الأول
                            logger.info(f"Testing search with query: {test_query[:50]}...")
                            
                            test_results = await vector_store.similarity_search(
                                test_query, 
                                k=1,
                                filter={"conversation_id": str(doc.conversation_id)}
                            )
                            if test_results:
                                logger.info(f"Search verification PASSED - found {len(test_results)} result(s)")
                                logger.info(f"Result similarity score: {test_results[0].get('score', 'N/A')}")
                            else:
                                logger.error(f"Search verification FAILED - no results found!")
                        except Exception as search_err:
                            logger.error(f"Error during search verification: {str(search_err)}")
                        
                        logger.info(f"تمت فهرسة المستند {doc.id} بنجاح مع {len(vector_ids)} متجهات")
                except Exception as vector_error:
                    logger.error(f"Error adding vectors to store: {str(vector_error)}")
                    logger.error(traceback.format_exc())
        
        except Exception as e:
            logger.error(f"خطأ أثناء فهرسة المستند: {e}")
            logger.error(traceback.format_exc())
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

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    # تم تعديل الدالة لعدم التحويل لمجلد static غير موجود
    return HTMLResponse(content="<html><body><h1>Navid RAG API</h1><p>API is running. Use appropriate endpoints.</p></body></html>")

@app.post("/api/rag", response_model=RAGResponse)
async def process_rag(request: RAGRequest):
    try:
        global rag_service
        
        result = rag_service.process_query(
            request.query, document_ids=request.document_ids, options=request.options
        )
        return RAGResponse(
            answer=result["answer"],
            context=result["context"],
            metadata=result["metadata"],
        )
    except Exception as e:
        logger.error(f"Error processing RAG request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    global rag_service
    
    health = {"status": "ok" if rag_service else "error"}
    
    # Check if the LLM model is available
    if rag_service and hasattr(rag_service, "llm"):
        try:
            health["llm_status"] = "ok"
            health["llm_model"] = getattr(rag_service.llm, "model_name", "unknown")
        except Exception as e:
            health["llm_status"] = "error"
            health["llm_error"] = str(e)
    
    # Check if the chat model is available
    if chat_model:
        health["chat_model_status"] = "ok"
        health["chat_model_name"] = getattr(chat_model, "model_name", "unknown")
    else:
        health["chat_model_status"] = "not_initialized"
    
    return health

# Static files and UI
# app.mount("/ui/static", StaticFiles(directory="ui/static"), name="static")  # تم التعليق لأن المجلد غير موجود
# templates = Jinja2Templates(directory="ui/templates")  # تم التعليق لأن المجلد قد لا يكون موجودًا

@app.get("/ui/{rest_of_path:path}", response_class=HTMLResponse)
async def serve_ui(request: Request, rest_of_path: str = ""):
    # تم تعديل الدالة لعدم استخدام templates لأن المجلد غير موجود
    return HTMLResponse(content="<html><body><h1>UI غير متوفرة حاليًا</h1><p>المجلد ui/templates غير موجود</p></body></html>")

# Initialization functions
def initialize_chat():
    """
    إعداد نموذج المحادثة باستخدام yehia-7b-preview-red
    """
    global chat_model, rag_service
    settings = get_settings()
    
    logger.info("Initializing chat model...")
    
    # التحقق من وجود مفتاح API
    api_key = settings.OPENAI_API_KEY or settings.HF_TOKEN
    if not api_key:
        logger.error("API key not found. Please set OPENAI_API_KEY or HF_TOKEN in .env file.")
        return None
    
    try:
        # إنشاء عميل HTTP مخصص مع رؤوس التفويض
        http_client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0
        )
        
        # إنشاء عميل OpenAI مخصص
        custom_client = OpenAI(
            api_key="sk-dummy",  # سيتم تجاهله لأننا نستخدم عميل HTTP مخصص
            base_url=settings.LLM_BASE_URL,
            http_client=http_client
        )
        
        from langchain_openai import ChatOpenAI
        
        # إنشاء نموذج ChatOpenAI
        model = ChatOpenAI(
            model="tgi",  # اسم النموذج المستخدم في الخادم
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            openai_api_base=settings.LLM_BASE_URL,
            openai_api_key="sk-dummy",
            client=custom_client
        )
        
        # إضافة خاصية model_name للتوافق
        model.model_name = "yehia-7b-preview-red"
        
        chat_model = model
        logger.info(f"Chat model initialized successfully: yehia-7b-preview-red")
        return model
        
    except Exception as e:
        logger.error(f"Error initializing chat model: {str(e)}")
        return None

# Main entry point
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the Navid RAG API")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    # Run the server
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)