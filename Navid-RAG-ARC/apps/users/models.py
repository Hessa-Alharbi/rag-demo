from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime, timezone
from sqlmodel import Field, SQLModel, Relationship
from apps.auth.utils import AuthUtils
from apps.organizations.models import Organization, UserOrganization


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str = Field(exclude=True)  # Exclude from response
    first_name: Optional[str] = Field(default=None)
    last_name: Optional[str] = Field(default=None)
    age: Optional[int] = Field(default=None)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationships
    organization_id: Optional[UUID] = Field(default=None, foreign_key="organization.id")
    organization: Optional[Organization] = Relationship(back_populates="users")
    organizations: List[Organization] = Relationship(
        back_populates="users",
        link_model=UserOrganization
    )
    conversations_created: List["Conversation"] = Relationship(  # type: ignore  # noqa: F821
        back_populates="created_by",
        sa_relationship_kwargs={"foreign_keys": "[Conversation.created_by_id]"}
    )
    documents: List["Document"] = Relationship(back_populates="created_by")  # type: ignore # Add this line  # noqa: F821

    def verify_password(self, password: str) -> bool:
        return AuthUtils.verify_password(password, self.hashed_password)

    @staticmethod
    def hash_password(password: str) -> str:
        return AuthUtils.get_password_hash(password)
