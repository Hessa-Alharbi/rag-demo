import json
from pydantic import BaseModel, validator
from typing import Optional
from uuid import UUID
from datetime import datetime
from apps.chat.models import MessageRole, MessageStatus, MessageType

class ConversationCreate(BaseModel):
    title: str

class ConversationRead(BaseModel):
    id: UUID
    title: str
    created_by_id: UUID
    created_at: datetime
    updated_at: datetime

class MessageCreate(BaseModel):
    content: str
    conversation_id: UUID
    role: MessageRole = MessageRole.USER
    type: MessageType = MessageType.TEXT
    message_metadata: dict = {}

class MessageRead(BaseModel):
    id: UUID
    content: str
    conversation_id: UUID
    user_id: UUID
    role: MessageRole
    type: MessageType
    status: MessageStatus
    message_metadata: dict
    created_at: datetime

    @validator('message_metadata', pre=True)
    def parse_metadata(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

class AttachmentRead(BaseModel):
    id: UUID
    filename: str
    file_type: str
    file_size: int
    conversation_id: UUID
    message_id: Optional[UUID] = None
    uploaded_by_id: UUID
    created_at: datetime
