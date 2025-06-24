from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from bson import ObjectId

from models.User import User, UserResponse, UserUpdate, FollowRequest
from models.Follow import Follow
from services.auth import get_current_user

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/me", response_model=UserResponse)
async def get_my_profile(current_user: User = Depends(get_current_user)):
    """
    Récupérer le profil de l'utilisateur connecté
    """
    return UserResponse(
        _id=str(current_user.id),
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
    current_user: User = Depends(get_current_user)
):
    """
    Mettre à jour le profil de l'utilisateur connecté
    """
    update_data = {}
    
    if user_update.username:
        # Vérifier que le nom d'utilisateur n'est pas déjà pris
        existing_user = await User.find_one(
            User.username == user_update.username,
            User.id != current_user.id
        )
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Ce nom d'utilisateur est déjà pris"
            )
        update_data["username"] = user_update.username
    
    if user_update.first_name is not None:
        update_data["first_name"] = user_update.first_name
    
    if user_update.last_name is not None:
        update_data["last_name"] = user_update.last_name
    if user_update.email:
        # Vérifier que l'email n'est pas déjà pris
        existing_user = await User.find_one(
            User.email == user_update.email,
            User.id != current_user.id
        )
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Cet email est déjà utilisé"
            )
        update_data["email"] = user_update.email
    
    if update_data:
        await current_user.update({"$set": update_data})
        await current_user.save()
    
    return UserResponse(
        _id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

@router.get("/suggestions", response_model=List[UserResponse])
async def get_suggested_users(
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir des suggestions d'utilisateurs à suivre
    """
    # Récupérer les IDs des utilisateurs que l'utilisateur actuel suit déjà
    following_docs = await Follow.find(Follow.follower_id == current_user.id).to_list()
    following_ids = [doc.followed_id for doc in following_docs]
    following_ids.append(current_user.id)  # Exclure l'utilisateur actuel
    
    # Trouver des utilisateurs que l'utilisateur actuel ne suit pas
    suggestions = await User.find(
        {"_id": {"$nin": following_ids}},
        limit=limit
    ).to_list()
    return [
        UserResponse(
            _id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at
        )
        for user in suggestions
    ]

@router.post("/follow/{user_id}")
async def follow_user(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Suivre un utilisateur
    """
    # Vérifier que l'utilisateur à suivre existe
    try:
        target_user = await User.get(ObjectId(user_id))
    except:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not target_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    # Vérifier que l'utilisateur ne se suit pas lui-même
    if str(target_user.id) == str(current_user.id):
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas vous suivre vous-même")
    
    # Vérifier si l'utilisateur ne suit pas déjà cet utilisateur
    existing_follow = await Follow.find_one(
        Follow.follower_id == current_user.id,
        Follow.followed_id == target_user.id
    )
    if existing_follow:
        raise HTTPException(status_code=400, detail="Vous suivez déjà cet utilisateur")
    
    # Créer la relation de suivi
    follow = Follow(
        follower_id=current_user.id,
        followed_id=target_user.id
    )
    await follow.insert()
    
    return {"message": "Utilisateur suivi avec succès"}

@router.delete("/unfollow/{user_id}")
async def unfollow_user(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Ne plus suivre un utilisateur
    """
    # Vérifier que l'utilisateur à ne plus suivre existe
    try:
        target_user = await User.get(ObjectId(user_id))
    except:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not target_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    # Trouver et supprimer la relation de suivi
    follow = await Follow.find_one(
        Follow.follower_id == current_user.id,
        Follow.followed_id == target_user.id
    )
    if not follow:
        raise HTTPException(status_code=400, detail="Vous ne suivez pas cet utilisateur")
    
    await follow.delete()
    
    return {"message": "Utilisateur retiré de vos abonnements avec succès"}

@router.get("/followers", response_model=List[UserResponse])
async def get_followers(
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir la liste des followers de l'utilisateur connecté
    """
    # Récupérer les IDs des followers
    follow_docs = await Follow.find(Follow.followed_id == current_user.id).to_list()
    follower_ids = [doc.follower_id for doc in follow_docs]
    
    if not follower_ids:
        return []
    
    # Récupérer les utilisateurs correspondants
    followers = await User.find({"_id": {"$in": follower_ids}}).to_list()
    
    return [
        UserResponse(
            _id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at
        )
        for user in followers
    ]

@router.get("/following", response_model=List[UserResponse])
async def get_following(
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir la liste des utilisateurs suivis par l'utilisateur connecté
    """
    # Récupérer les IDs des utilisateurs suivis
    follow_docs = await Follow.find(Follow.follower_id == current_user.id).to_list()
    following_ids = [doc.followed_id for doc in follow_docs]
    
    if not following_ids:
        return []
    
    # Récupérer les utilisateurs correspondants
    following = await User.find({"_id": {"$in": following_ids}}).to_list()
    
    return [
        UserResponse(
            _id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at
        )
        for user in following
    ]

@router.get("/{user_id}", response_model=UserResponse)
async def get_user_profile(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Récupérer le profil d'un utilisateur par son ID
    """
    try:
        user = await User.get(ObjectId(user_id))
    except:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    return UserResponse(
        _id=str(user.id),
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        created_at=user.created_at
    )

@router.get("/search/{query}", response_model=List[UserResponse])
async def search_users(
    query: str,
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    """
    Rechercher des utilisateurs par nom d'utilisateur, prénom ou nom
    """
    # Recherche par regex (insensible à la casse)
    users = await User.find(
        {
            "$or": [
                {"username": {"$regex": query, "$options": "i"}},
                {"first_name": {"$regex": query, "$options": "i"}},
                {"last_name": {"$regex": query, "$options": "i"}}
            ]
        },
        limit=limit
    ).to_list()
    
    return [
        UserResponse(
            _id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at
        )
        for user in users
    ]

@router.get("/{user_id}/follow-stats")
async def get_user_follow_stats(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir les statistiques de suivi d'un utilisateur
    """
    try:
        target_user = await User.get(ObjectId(user_id))
    except:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not target_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    # Compter les followers
    followers_count = await Follow.find(Follow.followed_id == target_user.id).count()
    
    # Compter les following
    following_count = await Follow.find(Follow.follower_id == target_user.id).count()
    
    return {
        "user_id": str(target_user.id),
        "followers_count": followers_count,
        "following_count": following_count
    }

@router.get("/{user_id}/is-following")
async def is_following_user(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Vérifier si l'utilisateur connecté suit un autre utilisateur
    """
    try:
        target_user = await User.get(ObjectId(user_id))
    except:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not target_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    # Vérifier si une relation de suivi existe
    follow = await Follow.find_one(
        Follow.follower_id == current_user.id,
        Follow.followed_id == target_user.id
    )
    
    return {
        "is_following": follow is not None,
        "follower_id": str(current_user.id),
        "followed_id": str(target_user.id)
    }

@router.get("/{user_id}/followers", response_model=List[UserResponse])
async def get_user_followers(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir la liste des followers d'un utilisateur spécifique
    """
    try:
        target_user = await User.get(ObjectId(user_id))
    except:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not target_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    # Récupérer les IDs des followers
    follow_docs = await Follow.find(Follow.followed_id == target_user.id).to_list()
    follower_ids = [doc.follower_id for doc in follow_docs]
    
    if not follower_ids:
        return []
    
    # Récupérer les utilisateurs correspondants
    followers = await User.find({"_id": {"$in": follower_ids}}).to_list()
    
    return [
        UserResponse(
            _id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at
        )
        for user in followers
    ]

@router.get("/{user_id}/following", response_model=List[UserResponse])
async def get_user_following(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Obtenir la liste des utilisateurs suivis par un utilisateur spécifique
    """
    try:
        target_user = await User.get(ObjectId(user_id))
    except:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    if not target_user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    # Récupérer les IDs des utilisateurs suivis
    follow_docs = await Follow.find(Follow.follower_id == target_user.id).to_list()
    following_ids = [doc.followed_id for doc in follow_docs]
    
    if not following_ids:
        return []
    
    # Récupérer les utilisateurs correspondants
    following = await User.find({"_id": {"$in": following_ids}}).to_list()
    
    return [
        UserResponse(
            _id=str(user.id),
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_active=user.is_active,
            created_at=user.created_at
        )
        for user in following
    ]
