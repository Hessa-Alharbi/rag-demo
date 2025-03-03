from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
from uuid import UUID


class UserLogin(BaseModel):
    username_or_email: str
    password: str


class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    exp: datetime
    type: str


class UserAuth(BaseModel):
    email: EmailStr
    password: str


class UserRegister(UserAuth):
    username: Optional[str] = None
    first_name: str
    last_name: str


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True

class RefreshTokenRequest(BaseModel):
    refresh_token: str
