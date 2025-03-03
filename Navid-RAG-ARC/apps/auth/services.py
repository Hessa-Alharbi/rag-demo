from fastapi import HTTPException, status
from sqlmodel import UUID, Session, select
from apps.users.models import User
from .utils import AuthUtils
from .schemas import UserRegister, UserLogin
import logging
from uuid import UUID

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, session: Session):
        self.session = session

    def _generate_unique_username(self, first_name: str, last_name: str) -> str:
        """Generate a unique username from first and last name"""
        base_username = f"{first_name.lower()}.{last_name.lower()}"
        username = base_username
        counter = 1

        while self.session.exec(select(User).where(User.username == username)).first():
            username = f"{base_username}{counter}"
            counter += 1

        return username

    async def register_user(self, user_data: UserRegister) -> User:
        """Register a new user"""
        try:
            # Check if email already exists
            existing_user = self.session.exec(
                select(User).where(User.email == user_data.email)
            ).first()

            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered",
                )

            # Generate username if not provided
            username = user_data.username
            if not username:
                username = self._generate_unique_username(
                    user_data.first_name, user_data.last_name
                )

            # Create new user
            hashed_password = User.hash_password(user_data.password)
            user = User(
                email=user_data.email,
                username=username,
                hashed_password=hashed_password,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
            )

            self.session.add(user)
            self.session.commit()
            self.session.refresh(user)
            return user

        except Exception as e:
            self.session.rollback()
            logger.error(f"Registration error: {str(e)}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed",
            )

    async def authenticate_user(self, login_data: UserLogin) -> User:
        """Authenticate user with either username or email"""
        try:
            # Try to find user by username or email
            user = self.session.exec(
                select(User).where(
                    (User.email == login_data.username_or_email)
                    | (User.username == login_data.username_or_email)
                )
            ).first()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            if not user.verify_password(login_data.password):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is inactive",
                )

            return user
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed",
            )

    async def create_tokens(self, user_id: str) -> dict:
        """Create access and refresh tokens"""
        try:
            return {
                "access_token": AuthUtils.create_access_token(user_id),
                "refresh_token": AuthUtils.create_refresh_token(user_id),
                "token_type": "bearer",
            }
        except Exception as e:
            logger.error(f"Token creation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create authentication tokens",
            )

    async def refresh_tokens(self, refresh_token: str) -> dict:
        """Refresh access token"""
        try:
            payload = AuthUtils.verify_refresh_token(refresh_token)
            user = self.session.exec(
                select(User).where(User.id == UUID(payload.sub))
            ).first()
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                )
                
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User is inactive",
                )
            
            # Convert UUID to string when creating tokens
            user_id_str = str(user.id)
            return {
                "access_token": AuthUtils.create_access_token(user_id_str),
                "refresh_token": AuthUtils.create_refresh_token(user_id_str),
                "token_type": "bearer"
            }
        except ValueError as e:
            logger.error(f"Token refresh error (invalid UUID): {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
