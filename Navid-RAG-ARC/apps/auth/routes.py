from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Session, select
from typing import Annotated
from uuid import UUID, uuid4

from core.db import get_session
from .schemas import TokenSchema, UserLogin, UserResponse, UserRegister, RefreshTokenRequest
from .services import AuthService
from .utils import AuthUtils
from apps.users.models import User

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# Add a test user for development/testing purposes
def get_current_user(session: Session = Depends(get_session)):
    # FOR TESTING PURPOSES ONLY - return a fake user
    # Check if test user exists
    test_user = session.exec(select(User).where(User.email == "test@example.com")).first()
    
    if not test_user:
        # Create test user
        test_user = User(
            id=uuid4(),
            email="test@example.com",
            username="test_user",
            hashed_password="DEVELOPMENT_MODE_NO_PASSWORD",
            is_active=True,
            first_name="Test",
            last_name="User"
        )
        session.add(test_user)
        session.commit()
        session.refresh(test_user)
    
    return test_user


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
