from typing import Optional, List, Dict, TYPE_CHECKING
from uuid import UUID, uuid4
from datetime import datetime, timezone
from sqlmodel import Field, SQLModel, Column, JSON, Relationship
from enum import Enum

if TYPE_CHECKING:
    from apps.chat.models import Conversation
    from apps.users.models import User

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    VECTORIZING = "vectorizing"
    COMPLETED = "completed"
    FAILED = "failed"

# Define Document class first
class Document(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    title: str
    content: str = Field(default="")
    doc_metadata: dict = Field(default={}, sa_column=Column(JSON))
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    vector_ids: List[str] = Field(default=[], sa_column=Column(JSON))
    conversation_id: Optional[UUID] = Field(default=None, foreign_key="conversation.id")
    created_by_id: UUID = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_status: DocumentProcessingStatus = Field(default=DocumentProcessingStatus.QUEUED)
    is_searchable: bool = Field(default=False)

    # Relationships
    conversation: Optional["Conversation"] = Relationship(back_populates="documents")
    created_by: Optional["User"] = Relationship(back_populates="documents")
    chunks: List["Chunk"] = Relationship(back_populates="document")
    processing_events: List["DocumentProcessingEvent"] = Relationship(back_populates="document")

class DocumentProcessingEvent(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="document.id")
    status: DocumentProcessingStatus
    message: Optional[str] = None
    event_metadata: Dict = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    document: Document = Relationship(back_populates="processing_events")

class Chunk(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="document.id")
    content: str
    chunk_metadata: dict = Field(default={}, sa_column=Column(JSON))
    vector_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    document: Document = Relationship(back_populates="chunks")
