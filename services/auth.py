from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
import secrets
import os
from dotenv import load_dotenv

from db.init_db import User
from models.User import UserCreate, UserLogin, UserResponse, Token

# Charger les variables d'environnement
load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Contexte de cryptage pour les mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Fonctions utilitaires pour les mots de passe
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifier un mot de passe en clair avec son hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hasher un mot de passe"""
    return pwd_context.hash(password)

# Fonctions de gestion des utilisateurs
async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Récupérer un utilisateur par email"""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()

async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Récupérer un utilisateur par nom d'utilisateur"""
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Récupérer un utilisateur par ID"""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """Créer un nouvel utilisateur"""
    # Vérifier que l'email et le nom d'utilisateur n'existent pas déjà
    existing_user = await get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Un utilisateur avec cet email existe déjà"
        )
    
    existing_username = await get_user_by_username(db, user_data.username)
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ce nom d'utilisateur est déjà pris"
        )
    
    # Créer le nouvel utilisateur
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        hashed_password=hashed_password,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """Authentifier un utilisateur"""
    user = await get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

# Fonctions de gestion des tokens JWT
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Créer un token d'accès JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Vérifier et décoder un token JWT"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None

async def get_current_user(db: AsyncSession, token: str) -> Optional[User]:
    """Récupérer l'utilisateur actuel à partir du token"""
    email = verify_token(token)
    if email is None:
        return None
    
    user = await get_user_by_email(db, email)
    return user

def generate_secret_key() -> str:
    """Générer une clé secrète sécurisée pour la production"""
    return secrets.token_urlsafe(32)