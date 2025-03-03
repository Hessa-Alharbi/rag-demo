from sqlmodel import SQLModel, Field
from datetime import datetime
from uuid import UUID, uuid4


class Token(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    token: str = Field(index=True)
    type: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_revoked: bool = Field(default=False)
