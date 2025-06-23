from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from beanie import Document
from bson import ObjectId
from .User import UserResponse

class Follow(Document):
    """Modèle de suivi entre utilisateurs"""
    follower_id: ObjectId  # ID de l'utilisateur qui suit
    followed_id: ObjectId  # ID de l'utilisateur suivi
    created_at: datetime = datetime.utcnow()
    
    class Settings:
        name = "follows"
        indexes = [
            ("follower_id", "followed_id"),  # Index composé pour les requêtes
            "follower_id",
            "followed_id"
        ]
    
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

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
