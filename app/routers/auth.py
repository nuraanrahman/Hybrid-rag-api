from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.db import get_db
from app.core.dependencies import get_current_user
from app.models.db import User
from app.models.schemas import UserCreate, Token

router = APIRouter(prefix="/auth", tags=["auth"])
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": exp}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

@router.post("/register", response_model=Token, status_code=201)
async def register(body: UserCreate, db: AsyncSession = Depends(get_db)):
    if (await db.execute(select(User).where(User.username == body.username))).scalar_one_or_none():
        raise HTTPException(400, "Username already taken")
    db.add(User(username=body.username, hashed_password=_pwd.hash(body.password)))
    await db.commit()
    return Token(access_token=_token(body.username))

@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.username == form.username))).scalar_one_or_none()
    if not user or not _pwd.verify(form.password, user.hashed_password):
        raise HTTPException(401, "Incorrect username or password")
    return Token(access_token=_token(user.username))

@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "username": user.username}
