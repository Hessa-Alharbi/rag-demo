from uuid import UUID
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str  # Plain password for registration
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None

class UserRead(BaseModel):
    id: UUID  # Changed from str to UUID
    username: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # Enable ORM mode
