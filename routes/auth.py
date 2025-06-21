from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from db.init_db import get_db
from models.User import UserCreate, UserLogin, UserResponse, Token
from services.auth import (
    authenticate_user,
    create_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

@router.post("/register", response_model=Token)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Enregistrer un nouvel utilisateur
    """
    try:
        # Créer l'utilisateur
        user = await create_user(db, user_data)
        
        # Créer le token d'accès
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        # Convertir en UserResponse pour la réponse
        user_response = UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # en secondes
            user=user_response
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de la création du compte"
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    Connecter un utilisateur existant
    """
    # Authentifier l'utilisateur
    user = await authenticate_user(db, user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Vérifier si le compte est actif
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Compte désactivé",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Créer le token d'accès
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Convertir en UserResponse pour la réponse
    user_response = UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # en secondes
        user=user_response
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Récupérer les informations de l'utilisateur connecté
    """
    user = await get_current_user(db, credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at
    )

@router.post("/logout")
async def logout():
    """
    Déconnecter un utilisateur (côté client, suppression du token)
    """
    return {"message": "Déconnexion réussie. Supprimez le token côté client."}

@router.post("/refresh", response_model=Token)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Renouveler le token d'accès
    """
    user = await get_current_user(db, credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Créer un nouveau token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    user_response = UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response
    )

# Dépendance pour récupérer l'utilisateur actuel dans d'autres routes
async def get_current_active_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Dépendance pour récupérer l'utilisateur actuel et vérifier qu'il est actif
    """
    user = await get_current_user(db, credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Compte inactif"
        )
    return user