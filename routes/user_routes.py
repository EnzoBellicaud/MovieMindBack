from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from db.init_db import get_db, User
from models.User import UserResponse, UserUpdate
from models.Follow import (
    FollowStatsResponse, 
    IsFollowingResponse, 
    MutualFollowsResponse,
    SuggestedUsersResponse
)
from routes.auth import get_current_active_user
from services.follow_service import FollowService

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user: User = Depends(get_current_active_user)):
    """
    Récupérer le profil de l'utilisateur connecté
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

@router.put("/me", response_model=UserResponse)
async def update_my_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Mettre à jour le profil de l'utilisateur connecté
    """
    # Vérifier si le nouveau nom d'utilisateur est déjà pris
    if user_update.username and user_update.username != current_user.username:
        existing_user = await db.execute(
            select(User).where(User.username == user_update.username)
        )
        if existing_user.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail="Ce nom d'utilisateur est déjà pris"
            )
    
    # Mettre à jour les champs
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    await db.commit()
    await db.refresh(current_user)
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

@router.get("/suggested")
async def get_suggested_users(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Obtenir des suggestions d'utilisateurs à suivre
    """
    suggestions = await FollowService.get_suggested_users(db, current_user.id, limit)
    return {
        "suggestions": [UserResponse(**user) for user in suggestions],
        "total": len(suggestions)
    }

@router.get("/mutual-follows/{user_id}", response_model=MutualFollowsResponse)
async def get_mutual_follows(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Vérifier les relations de suivi mutuelles avec un autre utilisateur
    """
    return await FollowService.get_mutual_follows(db, current_user.id, user_id)


@router.post("/{user_id}/follow")
async def follow_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Suivre un utilisateur
    """
    return await FollowService.follow_user(db, current_user.id, user_id)

@router.delete("/{user_id}/follow")
async def unfollow_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ne plus suivre un utilisateur
    """
    return await FollowService.unfollow_user(db, current_user.id, user_id)

@router.get("/{user_id}/followers", response_model=List[UserResponse])
async def get_user_followers(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupérer la liste des followers d'un utilisateur
    """
    followers = await FollowService.get_followers(db, user_id)
    return [UserResponse(**follower) for follower in followers]

@router.get("/{user_id}/following", response_model=List[UserResponse])
async def get_user_following(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupérer la liste des utilisateurs suivis par un utilisateur
    """
    following = await FollowService.get_following(db, user_id)
    return [UserResponse(**followed) for followed in following]

@router.get("/{user_id}/follow-stats", response_model=FollowStatsResponse)
async def get_user_follow_stats(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupérer les statistiques de suivi d'un utilisateur
    """
    return await FollowService.get_follow_stats(db, user_id)

@router.get("/{user_id}/is-following", response_model=IsFollowingResponse)
async def check_if_following(
    user_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Vérifier si l'utilisateur connecté suit un autre utilisateur
    """
    is_following = await FollowService.is_following(db, current_user.id, user_id)
    return IsFollowingResponse(
        is_following=is_following,
        follower_id=current_user.id,
        followed_id=user_id
    )
    
    
@router.get("/{user_id}", response_model=UserResponse)
async def get_user_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupérer le profil d'un utilisateur par son ID
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at
    )
