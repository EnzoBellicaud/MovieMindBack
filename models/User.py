from typing import Optional, List
from pydantic import BaseModel, EmailStr
from datetime import datetime

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
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

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