from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import Optional, Any, Dict
from enum import Enum

class ErrorCode(str, Enum):
    CONVERSATION_NOT_FOUND = "conversation_not_found"
    PERMISSION_DENIED = "permission_denied"
    INVALID_REQUEST = "invalid_request"
    DATABASE_ERROR = "database_error"
    ATTACHMENT_ERROR = "attachment_error"
    FILE_UPLOAD_ERROR = "file_upload_error"

class ErrorResponse(BaseModel):
    error: str
    message: str
    code: int
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None

    def __init__(self, **data):
        from datetime import datetime
        data["timestamp"] = datetime.utcnow().isoformat()
        super().__init__(**data)

class APIError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "internal_error",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }

class DatabaseError(APIError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="database_error",
            details={"original_error": str(original_error)} if original_error else None
        )

class ConversationNotFoundError(APIError):
    def __init__(self, conversation_id: str):
        super().__init__(
            message=f"Conversation {conversation_id} not found",
            status_code=404,
            error_code="conversation_not_found"
        )
