from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from beanie import Document
from datetime import datetime
from bson import ObjectId

# Beanie Document model for MongoDB
class User(Document):
    email: EmailStr = Field(..., unique=True)
    username: str = Field(..., unique=True)
    first_name: str
    last_name: str
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    movie_preferences: Optional[List[str]] = []
    
    following: List[str] = [] 
    followers: List[str] = []  # Store as strings instead of ObjectId
    
    # Liked movies for recommendations
    liked_movies: List[str] = []    # Store as strings instead of ObjectId
    disliked_movies: List[str] = []  # Store as strings instead of ObjectId
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "users"

# Pydantic models for API
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

class UserResponse(UserBase):
    id: str = Field(alias="_id")
    is_active: bool
    created_at: datetime
    following_count: int = 0
    followers_count: int = 0

    class Config:
        from_attributes = True
        populate_by_name = True

class UserUpdate(BaseModel):
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    movie_preferences: Optional[List[str]] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse

class FollowRequest(BaseModel):
    user_id: str