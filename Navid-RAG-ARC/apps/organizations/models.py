from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime, timezone
from sqlmodel import Field, SQLModel, Relationship

class OrganizationBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = None

class Organization(OrganizationBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Relationships
    users: List["User"] = Relationship(back_populates="organization")  # type: ignore # noqa: F821

class UserOrganization(SQLModel, table=True):
    user_id: UUID = Field(foreign_key="user.id", primary_key=True)
    organization_id: UUID = Field(foreign_key="organization.id", primary_key=True)
    role: str = Field(default="member")  # Could be: owner, admin, member
    joined_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
