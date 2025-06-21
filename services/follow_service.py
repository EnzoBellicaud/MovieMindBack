from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, insert, delete
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status

from db.init_db import User, follow_table


class FollowService:
    """Service pour gérer les relations de suivi entre utilisateurs"""
    
    @staticmethod
    async def follow_user(db: AsyncSession, follower_id: int, followed_id: int) -> dict:
        """
        Faire suivre un utilisateur par un autre
        """
        # Vérifier que l'utilisateur ne se suit pas lui-même
        if follower_id == followed_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vous ne pouvez pas vous suivre vous-même"
            )
        
        # Récupérer les utilisateurs
        follower = await db.get(User, follower_id)
        followed = await db.get(User, followed_id)
        
        if not follower or not followed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        # Vérifier si la relation existe déjà
        is_already_following = await FollowService.is_following(db, follower_id, followed_id)
        if is_already_following:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vous suivez déjà cet utilisateur"
            )
        
        # Ajouter la relation de suivi directement dans la table
        stmt = insert(follow_table).values(
            follower_id=follower_id,
            followed_id=followed_id
        )
        await db.execute(stmt)
        await db.commit()
        
        return {
            "message": f"Vous suivez maintenant {followed.username}",
            "follower": {
                "id": follower.id,
                "username": follower.username
            },
            "followed": {
                "id": followed.id,
                "username": followed.username
            }
        }
    
    @staticmethod
    async def unfollow_user(db: AsyncSession, follower_id: int, followed_id: int) -> dict:
        """
        Arrêter de suivre un utilisateur
        """
        # Récupérer les utilisateurs
        follower = await db.get(User, follower_id)
        followed = await db.get(User, followed_id)
        
        if not follower or not followed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        # Vérifier si la relation existe
        is_following = await FollowService.is_following(db, follower_id, followed_id)
        if not is_following:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vous ne suivez pas cet utilisateur"
            )
        
        # Supprimer la relation de suivi directement de la table
        stmt = delete(follow_table).where(
            and_(
                follow_table.c.follower_id == follower_id,
                follow_table.c.followed_id == followed_id
            )
        )
        await db.execute(stmt)
        await db.commit()
        
        return {
            "message": f"Vous ne suivez plus {followed.username}",
            "follower": {
                "id": follower.id,
                "username": follower.username
            },
            "unfollowed": {
                "id": followed.id,
                "username": followed.username
            }
        }
    
    @staticmethod
    async def is_following(db: AsyncSession, follower_id: int, followed_id: int) -> bool:
        """
        Vérifier si un utilisateur en suit un autre
        """
        result = await db.execute(
            select(follow_table).where(
                and_(
                    follow_table.c.follower_id == follower_id,
                    follow_table.c.followed_id == followed_id
                )
            )
        )
        return result.first() is not None
    
    @staticmethod
    async def get_followers(db: AsyncSession, user_id: int) -> List[dict]:
        """
        Récupérer la liste des followers d'un utilisateur
        """
        # Vérifier que l'utilisateur existe
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        # Récupérer les followers
        result = await db.execute(
            select(User)
            .join(follow_table, User.id == follow_table.c.follower_id)
            .where(follow_table.c.followed_id == user_id)
        )
        followers = result.scalars().all()
        
        return [
            {
                "id": follower.id,
                "username": follower.username,
                "first_name": follower.first_name,
                "last_name": follower.last_name,
                "email": follower.email,
                "is_active": follower.is_active,
                "created_at": follower.created_at
            }
            for follower in followers
        ]
    
    @staticmethod
    async def get_following(db: AsyncSession, user_id: int) -> List[dict]:
        """
        Récupérer la liste des utilisateurs suivis par un utilisateur
        """
        # Vérifier que l'utilisateur existe
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        # Récupérer les utilisateurs suivis
        result = await db.execute(
            select(User)
            .join(follow_table, User.id == follow_table.c.followed_id)
            .where(follow_table.c.follower_id == user_id)
        )
        following = result.scalars().all()
        
        return [
            {
                "id": followed.id,
                "username": followed.username,
                "first_name": followed.first_name,
                "last_name": followed.last_name,
                "email": followed.email,
                "is_active": followed.is_active,
                "created_at": followed.created_at
            }
            for followed in following
        ]
    
    @staticmethod
    async def get_follow_stats(db: AsyncSession, user_id: int) -> dict:
        """
        Récupérer les statistiques de suivi d'un utilisateur
        """
        # Vérifier que l'utilisateur existe
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Utilisateur introuvable"
            )
        
        # Compter les followers
        followers_result = await db.execute(
            select(follow_table).where(follow_table.c.followed_id == user_id)
        )
        followers_count = len(followers_result.fetchall())
        
        # Compter les utilisateurs suivis
        following_result = await db.execute(
            select(follow_table).where(follow_table.c.follower_id == user_id)
        )
        following_count = len(following_result.fetchall())
        
        return {
            "user_id": user_id,
            "username": user.username,
            "followers_count": followers_count,
            "following_count": following_count
        }
    
    @staticmethod
    async def get_mutual_follows(db: AsyncSession, user1_id: int, user2_id: int) -> dict:
        """
        Vérifier les relations de suivi mutuelles entre deux utilisateurs
        """
        user1_follows_user2 = await FollowService.is_following(db, user1_id, user2_id)
        user2_follows_user1 = await FollowService.is_following(db, user2_id, user1_id)
        
        return {
            "user1_follows_user2": user1_follows_user2,
            "user2_follows_user1": user2_follows_user1,
            "mutual_follow": user1_follows_user2 and user2_follows_user1
        }
    
    @staticmethod
    async def get_suggested_users(db: AsyncSession, user_id: int, limit: int = 10) -> List[dict]:
        """
        Suggérer des utilisateurs à suivre basé sur les utilisateurs non suivis
        """
        # Récupérer les IDs des utilisateurs déjà suivis
        following_result = await db.execute(
            select(follow_table.c.followed_id).where(follow_table.c.follower_id == user_id)
        )
        following_ids = {row[0] for row in following_result.fetchall()}
        following_ids.add(user_id)  # Exclure l'utilisateur lui-même
        
        # Récupérer les utilisateurs non suivis
        result = await db.execute(
            select(User)
            .where(~User.id.in_(following_ids))
            .where(User.is_active == True)
            .limit(limit)
        )
        suggested_users = result.scalars().all()
        
        return [
            {
                "id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "email": user.email,
                "is_active": user.is_active,
                "created_at": user.created_at
            }
            for user in suggested_users
        ]
