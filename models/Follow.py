from typing import List
from pydantic import BaseModel
from datetime import datetime
from .User import UserResponse

class FollowResponse(BaseModel):
    """Réponse pour une action de suivi"""
    message: str
    follower: dict
    followed: dict

class UnfollowResponse(BaseModel):
    """Réponse pour une action de ne plus suivre"""
    message: str
    follower: dict
    unfollowed: dict

class FollowStatsResponse(BaseModel):
    """Statistiques de suivi d'un utilisateur"""
    user_id: int
    username: str
    followers_count: int
    following_count: int

class IsFollowingResponse(BaseModel):
    """Réponse pour vérifier si un utilisateur en suit un autre"""
    is_following: bool
    follower_id: int
    followed_id: int

class MutualFollowsResponse(BaseModel):
    """Réponse pour les relations de suivi mutuelles"""
    user1_follows_user2: bool
    user2_follows_user1: bool
    mutual_follow: bool

class SuggestedUsersResponse(BaseModel):
    """Réponse pour les suggestions d'utilisateurs"""
    suggestions: List[UserResponse]
    total: int

class UserSummary(BaseModel):
    """Résumé d'un utilisateur pour les listes"""
    id: int
    username: str
    first_name: str
    last_name: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
