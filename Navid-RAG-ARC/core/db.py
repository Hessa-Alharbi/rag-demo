from sqlmodel import SQLModel, Session, create_engine
from core.config import get_settings
from loguru import logger
from contextlib import contextmanager
from typing import Generator
from .errors import DatabaseError

# Import all models to ensure they're registered with proper order
from apps.users.models import User
from apps.organizations.models import Organization, UserOrganization
from apps.rag.models import Document, Chunk
from apps.chat.models import Conversation, Message, Attachment

settings = get_settings()

DATABASE_URL = "sqlite:///./sql_app.db"
engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    logger.info("Initializing database...")
    try:
        # This will create all tables and establish relationships
        SQLModel.metadata.create_all(engine)
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise

def get_session():
    """
    FastAPI dependency that creates a new database session.
    Unlike get_db, this is not a context manager and should be used with FastAPI Depends.
    """
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = Session(engine)
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise DatabaseError("Database operation failed", original_error=e)
    finally:
        db.close()
