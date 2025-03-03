from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone
from sqlmodel import Field, SQLModel, Relationship, Column, JSON
from enum import Enum
from apps.users.models import User

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class MessageType(str, Enum):
    TEXT = "text"
    COMMAND = "command"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESULT = "function_result"

class Conversation(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    title: str
    created_by_id: UUID = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    messages: List["Message"] = Relationship(back_populates="conversation")
    created_by: User = Relationship(back_populates="conversations_created")
    attachments: List["Attachment"] = Relationship(back_populates="conversation")
    documents: List["Document"] = Relationship(back_populates="conversation")  # type: ignore # noqa: F821

class Message(SQLModel, table=True):
    __tablename__ = "message"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    content: str
    conversation_id: UUID = Field(foreign_key="conversation.id")
    user_id: UUID = Field(foreign_key="user.id")
    role: Optional[str] = Field(default="user", nullable=False)
    type: Optional[str] = Field(default="text", nullable=False)
    status: Optional[str] = Field(default="completed", nullable=False)
    message_metadata: Optional[dict] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    conversation: Conversation = Relationship(back_populates="messages")
    user: User = Relationship()

class Attachment(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    filename: str
    file_path: str
    file_type: str
    file_size: int
    conversation_id: UUID = Field(foreign_key="conversation.id")
    message_id: Optional[UUID] = Field(default=None, foreign_key="message.id")
    uploaded_by_id: UUID = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    conversation: "Conversation" = Relationship(back_populates="attachments")
    uploaded_by: User = Relationship()
    message: Optional["Message"] = Relationship()
