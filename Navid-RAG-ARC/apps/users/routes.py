from fastapi import APIRouter, Depends
from typing import Annotated, List
from sqlmodel import Session, select
from apps.users.models import User
from apps.users.schemas import UserRead
from core.db import get_session
from apps.auth.routes import get_current_user

router = APIRouter()


@router.get("/", response_model=List[UserRead])
async def read_users(
    current_user: Annotated[User, Depends(get_current_user)],  # Require authentication
    session: Session = Depends(get_session),
):
    users = session.exec(select(User)).all()
    return users


@router.get("/me", response_model=UserRead)
async def get_my_profile(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user
