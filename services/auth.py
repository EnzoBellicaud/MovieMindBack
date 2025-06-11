from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import secrets

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Contexte de cryptage pour les mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Modèles Pydantic
class UserBase(BaseModel):
    email: EmailStr
    username: str
    first_name: str
    last_name: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: User

class TokenData(BaseModel):
    email: Optional[str] = None

# Base de données mockée (simulation)
fake_users_db: Dict[str, UserInDB] = {
    "john.doe@example.com": UserInDB(
        id=1,
        email="john.doe@example.com",
        username="johndoe",
        first_name="John",
        last_name="Doe",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        is_active=True,
        created_at=datetime.now()
    ),
    "jane.smith@example.com": UserInDB(
        id=2,
        email="jane.smith@example.com",
        username="janesmith",
        first_name="Jane",
        last_name="Smith",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        is_active=True,
        created_at=datetime.now()
    ),
    "admin@moviemind.com": UserInDB(
        id=3,
        email="admin@moviemind.com",
        username="admin",
        first_name="Admin",
        last_name="MovieMind",
        hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        is_active=True,
        created_at=datetime.now()
    )
}

# Fonctions utilitaires pour les mots de passe
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifier un mot de passe en clair avec son hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hasher un mot de passe"""
    return pwd_context.hash(password)

# Fonctions de gestion des utilisateurs (mockées)
def get_user(email: str) -> Optional[UserInDB]:
    """Récupérer un utilisateur par email depuis la DB mockée"""
    return fake_users_db.get(email)

def get_user_by_username(username: str) -> Optional[UserInDB]:
    """Récupérer un utilisateur par nom d'utilisateur depuis la DB mockée"""
    for user in fake_users_db.values():
        if user.username == username:
            return user
    return None

def create_user(user_data: UserCreate) -> UserInDB:
    """Créer un nouvel utilisateur dans la DB mockée"""
    if get_user(user_data.email) or get_user_by_username(user_data.username):
        raise ValueError("Un utilisateur avec cet email ou ce nom d'utilisateur existe déjà")
    
    # Créer le nouvel utilisateur
    new_user = UserInDB(
        id=len(fake_users_db) + 1,
        email=user_data.email,
        username=user_data.username,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        hashed_password=get_password_hash(user_data.password),
        is_active=True,
        created_at=datetime.now()
    )
    
    # L'ajouter à la DB mockée
    fake_users_db[user_data.email] = new_user
    return new_user

def authenticate_user(email: str, password: str) -> Optional[UserInDB]:
    """Authentifier un utilisateur"""
    user = get_user(email)
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

def verify_token(token: str) -> Optional[TokenData]:
    """Vérifier et décoder un token JWT"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        token_data = TokenData(email=email)
        return token_data
    except JWTError:
        return None

def get_current_user(token: str) -> Optional[User]:
    """Récupérer l'utilisateur actuel à partir du token"""
    token_data = verify_token(token)
    if token_data is None:
        return None
    
    user = get_user(email=token_data.email)
    if user is None:
        return None
    
    # Convertir UserInDB en User (sans le mot de passe hashé)
    return User(
        id=user.id,
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at
    )

def generate_secret_key() -> str:
    """Générer une clé secrète sécurisée pour la production"""
    return secrets.token_urlsafe(32)