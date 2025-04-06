from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.db import init_db
from core.logger import logger
from core.config import get_settings  # Changed from core.settings to core.config
from core.middleware import error_handler
from core.vector_store.connection import MilvusConnectionManager
from core.startup import initialize_vector_stores

from apps.admin.routes import router as admin_router
from apps.auth.routes import router as auth_router
from apps.chat.routes import router as chat_router
from apps.organizations.routes import router as org_router
from apps.test.routes import router as test_router


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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)