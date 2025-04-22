from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from passlib.context import CryptContext
from fastapi import HTTPException, status
from .schemas import TokenPayload
from jose import jwt 
import logging
from uuid import UUID

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = "a-random-and-unsecure-key-just-for-testing"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthUtils:
    @staticmethod
    def _create_token(subject: str, expires_delta: timedelta, token_type: str) -> str:
        try:
            # Convert UUID to string if needed
            subject_str = str(subject) if isinstance(subject, UUID) else subject
            now = datetime.now(timezone.utc)
            expire = now + expires_delta
            to_encode: Dict[str, Any] = {
                "sub": subject_str,
                "exp": expire,
                "type": token_type,
                "iat": now,
            }
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Token creation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create token",
            )

    @classmethod
    def create_access_token(cls, subject: str) -> str:
        return cls._create_token(
            subject, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), "access"
        )

    @classmethod
    def create_refresh_token(cls, subject: str) -> str:
        return cls._create_token(
            subject, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS), "refresh"
        )

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_token(token: str) -> TokenPayload:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            exp_datetime = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)

            if exp_datetime < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                )

            # Ensure sub is a valid UUID string
            try:
                UUID(payload["sub"])
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token subject",
                )

            return TokenPayload(
                sub=payload["sub"], exp=exp_datetime, type=payload["type"]
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not validate credentials: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            )

    @classmethod
    def verify_refresh_token(cls, token: str) -> TokenPayload:
        token_data = cls.verify_token(token)
        if token_data.type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )
        return token_data
