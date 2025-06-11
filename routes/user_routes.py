from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.types import JSON
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from db.init_db import get_session, User
from models.User import UserCreate, UserRead

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)


@router.post("/", response_model=UserRead)
async def create_user(user: UserCreate, session: AsyncSession = Depends(get_session)):
    db_user = User(**user.dict())
    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    return db_user

@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: int, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/{user_id}/follow/{target_id}")
async def follow_user(user_id: int, target_id: int, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    target = await session.get(User, target_id)
    if not user or not target:
        raise HTTPException(status_code=404, detail="User or target not found")
    if target in user.following:
        raise HTTPException(status_code=400, detail="Already following")
    user.following.append(target)
    await session.commit()
    return {"message": f"User {user_id} is now following User {target_id}"}

@router.get("/{user_id}/followers", response_model=List[UserRead])
async def get_followers(user_id: int, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.followers

@router.get("/{user_id}/following", response_model=List[UserRead])
async def get_following(user_id: int, session: AsyncSession = Depends(get_session)):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user.following