from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Session, select
from typing import Annotated
from uuid import UUID

from core.db import get_session
from .schemas import TokenSchema, UserLogin, UserResponse, UserRegister, RefreshTokenRequest
from .services import AuthService
from .utils import AuthUtils
from apps.users.models import User

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Session = Depends(get_session),
) -> User:
    try:
        payload = AuthUtils.verify_token(token)
        user = session.exec(
            select(User).where(User.id == UUID(payload.sub), User.is_active.is_(True))
        ).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister, session: Session = Depends(get_session)):
    auth_service = AuthService(session)
    return await auth_service.register_user(user_data)


@router.post("/login", response_model=TokenSchema)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Session = Depends(get_session),
):
    try:
        auth_service = AuthService(session)
        user = await auth_service.authenticate_user(
            UserLogin(username_or_email=form_data.username, password=form_data.password)
        )
        return await auth_service.create_tokens(str(user.id))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        )


@router.post("/refresh", response_model=TokenSchema)
async def refresh_token(
    token_data: RefreshTokenRequest,
    session: Session = Depends(get_session)
):
    auth_service = AuthService(session)
    return await auth_service.refresh_tokens(token_data.refresh_token)


@router.get("/me", response_model=UserResponse)
async def get_user_profile(current_user: Annotated[User, Depends(get_current_user)]):
    return current_user
