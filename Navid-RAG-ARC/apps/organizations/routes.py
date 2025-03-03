from fastapi import APIRouter, Depends
from typing import List
from sqlmodel import Session, select
from apps.organizations.models import Organization, UserOrganization
from apps.organizations.schemas import OrganizationCreate, OrganizationRead
from apps.users.models import User
from core.db import get_session
from apps.auth.routes import get_current_user

router = APIRouter()

@router.post("/", response_model=OrganizationRead)
async def create_organization(
    org_data: OrganizationCreate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    org = Organization(**org_data.dict())
    session.add(org)
    
    # Create user-organization relationship with owner role
    user_org = UserOrganization(
        user_id=current_user.id,
        organization_id=org.id,
        role="owner"
    )
    session.add(user_org)
    
    session.commit()
    session.refresh(org)
    return org

@router.get("/", response_model=List[OrganizationRead])
async def list_organizations(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    query = select(Organization).join(UserOrganization).where(
        UserOrganization.user_id == current_user.id
    )
    orgs = session.exec(query).all()
    return orgs
