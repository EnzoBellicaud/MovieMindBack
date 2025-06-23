from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import os
from dotenv import load_dotenv

from models.User import User, UserCreate, UserLogin, UserResponse, Token

# Charger les variables d'environnement
load_dotenv()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Contexte de cryptage pour les mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme for JWT
security = HTTPBearer()

# Fonctions utilitaires pour les mots de passe
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifier un mot de passe en clair avec son hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hasher un mot de passe"""
    return pwd_context.hash(password)

# Fonctions de gestion des utilisateurs avec MongoDB
async def get_user_by_email(email: str) -> Optional[User]:
    """Récupérer un utilisateur par email"""
    return await User.find_one(User.email == email)

async def get_user_by_username(username: str) -> Optional[User]:
    """Récupérer un utilisateur par nom d'utilisateur"""
    return await User.find_one(User.username == username)

async def get_user_by_id(user_id: str) -> Optional[User]:
    """Récupérer un utilisateur par ID"""
    return await User.get(user_id)

async def create_user(user_data: UserCreate) -> User:
    """Créer un nouvel utilisateur"""
    # Vérifier que l'email et le nom d'utilisateur n'existent pas déjà
    existing_user = await get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Un utilisateur avec cet email existe déjà"
        )
    
    existing_username = await get_user_by_username(user_data.username)
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
    
    await db_user.insert()
    return db_user

async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authentifier un utilisateur"""
    user = await get_user_by_email(email)
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

async def get_current_user_from_token(token: str) -> Optional[User]:
    """Récupérer l'utilisateur actuel à partir du token"""
    email = verify_token(token)
    if email is None:
        return None
    
    user = await get_user_by_email(email)
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Dependency pour récupérer l'utilisateur actuel"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        user = await get_current_user_from_token(credentials.credentials)
        if user is None:
            raise credentials_exception
        return user
    except Exception:
        raise credentials_exception

def generate_secret_key() -> str:
    """Générer une clé secrète sécurisée pour la production"""
    return secrets.token_urlsafe(32)
