from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

class OrganizationCreate(BaseModel):
    name: str
    description: Optional[str] = None

class OrganizationRead(OrganizationCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
