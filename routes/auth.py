from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta

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
async def register(user_data: UserCreate):
    """
    Enregistrer un nouvel utilisateur
    """
    try:
        # Créer l'utilisateur
        user = await create_user(user_data)
        
        # Créer le token d'accès
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        
        # Convertir en UserResponse pour la réponse
        user_response = UserResponse(
            _id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at,
            following_count=len(user.following),
            followers_count=len(user.followers)
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
async def login(user_credentials: UserLogin):
    """
    Connecter un utilisateur existant
    """
    # Authentifier l'utilisateur
    user = await authenticate_user(user_credentials.email, user_credentials.password)
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
        _id=str(user.id),
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at,
        following_count=len(user.following),
        followers_count=len(user.followers)
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # en secondes
        user=user_response
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Récupérer les informations de l'utilisateur connecté
    """
    return UserResponse(
        _id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        following_count=len(current_user.following),
        followers_count=len(current_user.followers)
    )

@router.post("/logout")
async def logout():
    """
    Déconnecter un utilisateur (côté client, suppression du token)
    """
    return {"message": "Déconnexion réussie. Supprimez le token côté client."}

@router.post("/refresh", response_model=Token)
async def refresh_token(current_user = Depends(get_current_user)):
    """
    Renouveler le token d'accès
    """
    # Créer un nouveau token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.email}, expires_delta=access_token_expires
    )
    
    user_response = UserResponse(
        _id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        following_count=len(current_user.following),
        followers_count=len(current_user.followers)
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response
    )

# Dépendance pour récupérer l'utilisateur actuel dans d'autres routes
async def get_current_active_user(current_user = Depends(get_current_user)):
    """
    Dépendance pour récupérer l'utilisateur actuel et vérifier qu'il est actif
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Compte inactif"
        )
    return current_user
