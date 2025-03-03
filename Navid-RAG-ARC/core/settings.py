# Move contents from core/config.py to core/settings.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache
from langchain.embeddings import OpenAIEmbeddings

class Settings(BaseSettings):
    # Base settings
    APP_NAME: str = "NavidRAG"
    DEBUG: bool = True
    API_PREFIX: str = "/api"
    
    # OpenAI settings
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_TEMPERATURE: float = 0.0

    # Database settings
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    
    # File upload settings
    UPLOAD_DIR: Path = Path("uploads")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Vector store settings
    VECTOR_STORE_DIR: Path = Path("vector_store")
    
    # Text processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # JWT settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def get_embeddings(self):
        """Get OpenAI embeddings instance"""
        return OpenAIEmbeddings(
            model=self.OPENAI_EMBEDDING_MODEL,
            openai_api_key=self.OPENAI_API_KEY
        )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
