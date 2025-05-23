import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "NavidRAG"
    DEBUG: bool = True
    API_PREFIX: str = "/api"
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"
    
    # LLM Configuration
    LLM_PROVIDER: str = ""  # or "huggingface", "anthropic", etc.
    LLM_MODEL: str = ""
    LLM_BASE_URL: str = ""
    LLM_API_KEY: Optional[str] = None
    LLM_CONFIG: Dict[str, Any] = {}
    REQUIRE_LLM: bool = True  # إجبار هذا الإعداد على True دائمًا لمنع استخدام النص المستخرج مباشرة
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None  # HuggingFace token
    
    # OpenAI specific settings
    OPENAI_MODEL: str = "tgi"  # موديل yehia-7b-preview-red ولكن بالاسم المتوافق مع خادم TGI
    OPENAI_MODEL_NAME: str = "tgi"  # موديل yehia-7b-preview-red ولكن بالاسم المتوافق مع خادم TGI
    OPENAI_EMBEDDING_MODEL: str = ""
    OPENAI_MAX_TOKENS: int = 800
    OPENAI_TEMPERATURE: float = 0.3
    
    # Embeddings Configuration
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_CONFIG: Dict[str, Any] = {
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True}
    }
    EMBEDDING_DIMENSION: int = 1536  # For OpenAI embeddings
    
    # Vector Store Configuration
    VECTOR_STORE_PROVIDER: str = "milvus"
    VECTOR_STORE_TYPE: str = "faiss"  # or "milvus", "qdrant", etc.
    
    # Milvus specific settings
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "document_store"
    MILVUS_AUTO_ID: bool = True
    MILVUS_DIMENSION: int = 384
    MILVUS_INDEX_TYPE: str = "IVF_FLAT"
    MILVUS_METRIC_TYPE: str = "L2"
    MILVUS_INDEX_PARAMS: Dict[str, Any] = {
        "metric_type": "L2"
    }
    MILVUS_SEARCH_PARAMS: Dict[str, Any] = {
        "metric_type": "L2"
    }
    
    # Chunking Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNK_MODEL: str = "recursive"  # or "markdown", "nltk", etc.
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    
    # JWT settings
    JWT_SECRET_KEY: str = "your-secret-key"  # Should be overridden in production
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True
    VECTOR_STORE_PATH: str = "./data/vector_store/faiss_index"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # هام: السماح بالحقول الإضافية للاستيعاب التوافقي مع ملف .env
        extra = "allow"
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        for path in [self.DATA_DIR, self.UPLOAD_DIR, self.VECTOR_STORE_DIR]:
            path.mkdir(parents=True, exist_ok=True)
    
    @property
    def milvus_connection_args(self) -> Dict[str, Any]:
        return {
            "host": self.MILVUS_HOST,
            "port": self.MILVUS_PORT
        }
    
    def validate_settings(self):
        """Validate required settings are present"""
        # تحقق أولاً من نموذج اللغة - منع استخدام Mistral
        if self.LLM_MODEL != "yehia-7b-preview-red":
            from loguru import logger
            logger.warning(f"لم يتم تحديد نموذج yehia-7b-preview-red، يرجى استخدام هذا النموذج فقط.")
            self.LLM_MODEL = "yehia-7b-preview-red"
            logger.info(f"تم تغيير LLM_MODEL إلى: {self.LLM_MODEL}")
        
        # منع استخدام api-inference.huggingface.co
        if "api-inference.huggingface.co" in self.LLM_BASE_URL:
            from loguru import logger
            logger.error(f"تم اكتشاف استخدام api-inference.huggingface.co، وهذا غير مسموح.")
            self.LLM_BASE_URL = "https://ijt42iqbf30i3nly.us-east4.gcp.endpoints.huggingface.cloud/v1"
            logger.info(f"تم تغيير LLM_BASE_URL إلى: {self.LLM_BASE_URL}")
        
        # تأكد من وجود حقل OPENAI_MODEL صحيح
        if self.OPENAI_MODEL != "tgi":
            from loguru import logger
            logger.warning(f"OPENAI_MODEL يجب أن يكون 'tgi' للعمل مع yehia-7b-preview-red")
            self.OPENAI_MODEL = "tgi"
            logger.info(f"تم تصحيح OPENAI_MODEL إلى: {self.OPENAI_MODEL}")
        
        if self.VECTOR_STORE_PROVIDER == "milvus":
            if not self.MILVUS_HOST or not self.MILVUS_PORT:
                raise ValueError("Milvus host and port must be configured")
            if not self.MILVUS_DIMENSION:
                raise ValueError("Milvus dimension must be configured based on embedding model")
        
        if self.EMBEDDING_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key must be configured when using OpenAI embeddings")
        
        if self.EMBEDDING_PROVIDER == "huggingface":
            if not self.EMBEDDING_MODEL:
                raise ValueError("Embedding model must be configured when using HuggingFace")
        
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key must be configured when using OpenAI LLM")
            
        # Validate paths exist
        for path in [self.DATA_DIR, self.UPLOAD_DIR, self.VECTOR_STORE_DIR]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    def get_embeddings(self):
        """Get configured embeddings model"""
        if self.EMBEDDING_PROVIDER == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL,
                **self.EMBEDDING_CONFIG
            )
        elif self.EMBEDDING_PROVIDER == "openai":
            return OpenAIEmbeddings(
                model=self.OPENAI_EMBEDDING_MODEL,
                openai_api_key=self.OPENAI_API_KEY
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.EMBEDDING_PROVIDER}")

@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_settings()
    return settings
