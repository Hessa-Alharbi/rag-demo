from pydantic import BaseModel
from typing import Optional
from enum import Enum

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentStatusResponse(BaseModel):
    id: str
    status: DocumentStatus
    filename: str
    error: Optional[str] = None
