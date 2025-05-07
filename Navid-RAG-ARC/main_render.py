"""
نسخة معدلة من تطبيق Navid RAG API مع تحسينات لاستخدام الذاكرة على Render
"""
import sys
import os
import gc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Body, Depends, status, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Any
import json
import traceback
from uuid import uuid4
import logging

# تكوين التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("navid-rag")

# تحديد الحد الأقصى لاستخدام الذاكرة (بالميجابايت)
MAX_MEMORY_USAGE = 450  # استخدم 450 من أصل 512 MB المتاحة في الخطة المجانية


# تنظيف الذاكرة يدوياً
def clean_memory():
    """تحرير الذاكرة غير المستخدمة يدوياً"""
    gc.collect()
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("Memory cleanup performed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application with memory optimizations...")
    
    try:
        # Import these here to avoid loading them until needed
        from core.db import init_db
        from core.vector_store.connection import MilvusConnectionManager
        from apps.rag.services import RAGService
        
        # تهيئة قاعدة البيانات
        init_db()
        logger.info("Database initialized")
        
        # إدارة الاتصال بـ Milvus
        MilvusConnectionManager.ensure_connection()
        logger.info("Milvus connection established")
        
        # تهيئة خدمة RAG بشكل تدريجي لتقليل استخدام الذاكرة
        rag_service = RAGService()
        # تهيئة بدون تحميل النماذج الكبيرة مسبقاً (سيتم تحميلها عند الحاجة)
        await rag_service.initialize(preload_models=False)
        app.state.rag_service = rag_service
        logger.info("RAG service initialized with lazy loading")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup
        if hasattr(app.state, 'rag_service'):
            del app.state.rag_service
        MilvusConnectionManager.close_connection()
        clean_memory()
        logger.info("Application shutdown complete")


# إنشاء تطبيق FastAPI مع إدارة دورة الحياة
app = FastAPI(lifespan=lifespan, title="Navid RAG API", root_path="/api")

# إضافة وسيط CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكن تغييرها لاستخدام عناوين محددة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إضافة وسيط لتنظيف الذاكرة بعد كل طلب
@app.middleware("http")
async def memory_cleanup_middleware(request: Request, call_next):
    response = await call_next(request)
    clean_memory()
    return response

# استيراد طرق التوجيه للتطبيق
from apps.auth.routes import router as auth_router
from apps.chat.routes import router as chat_router
from apps.organizations.routes import router as org_router
from apps.rag.routes import router as rag_router

# تسجيل طرق التوجيه
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(chat_router, prefix="/conversations", tags=["conversations"])
app.include_router(org_router, prefix="/organizations", tags=["organizations"])
app.include_router(rag_router, prefix="/rag", tags=["rag"])

# فحص تشغيل التطبيق
@app.get("/health")
async def health_check():
    """نقطة نهاية للتحقق من أن التطبيق يعمل بشكل صحيح"""
    return {"status": "ok", "service": "Navid RAG API"}

# توجيه الجذر
@app.get("/", response_class=HTMLResponse)
async def root():
    """الصفحة الرئيسية للتطبيق"""
    return """
    <html>
        <head>
            <title>Navid RAG API</title>
        </head>
        <body>
            <h1>Navid RAG API</h1>
            <p>API is running. Please use the appropriate endpoints.</p>
        </body>
    </html>
    """

# إضافة نقطة نهاية محسنة لمعالجة RAG على Render
@app.post("/rag/query", response_model=Dict[str, Any])
async def process_rag_optimized(query: Dict[str, Any] = Body(...)):
    """
    نقطة نهاية محسنة للذاكرة لمعالجة استعلامات RAG
    تقسم العملية إلى خطوات أصغر لتجنب استخدام الكثير من الذاكرة
    """
    # تنظيف الذاكرة قبل المعالجة
    clean_memory()
    
    try:
        # الوصول إلى خدمة RAG المُهيأة سابقاً
        rag_service = app.state.rag_service
        
        # معالجة الاستعلام مع تقييد الذاكرة
        response = await rag_service.process_query(
            query["text"], 
            max_sources=3,  # تقليل عدد المصادر للحفاظ على الذاكرة
            stream=False
        )
        
        # تنظيف الذاكرة بعد المعالجة
        clean_memory()
        
        return response
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # بدء التشغيل مع إعدادات محسنة للذاكرة
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        workers=1,
        limit_concurrency=4,
        timeout_keep_alive=30
    ) 