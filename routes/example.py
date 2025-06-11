from fastapi import APIRouter, Depends
from .auth import get_current_active_user
from services.auth import User

router = APIRouter(
    prefix="/protected",
    tags=["protected"],
    responses={404: {"description": "Not found"}},
)

@router.get("/profile", tags=["profile"])
async def get_user_profile(current_user: User = Depends(get_current_active_user)):
    """
    Exemple de route protégée - nécessite un token valide
    """
    return {
        "message": f"Bonjour {current_user.first_name} {current_user.last_name}!",
        "user_info": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "is_active": current_user.is_active,
            "created_at": current_user.created_at
        }
    }

@router.get("/movies/recommendations", tags=["movies"])
async def get_movie_recommendations(current_user: User = Depends(get_current_active_user)):
    """
    Exemple de route protégée pour les recommandations de films
    """
    # Ici on pourrait récupérer les préférences de l'utilisateur depuis la DB
    # et générer des recommandations personnalisées
    mock_recommendations = [
        {
            "id": 1,
            "title": "The Matrix",
            "year": 1999,
            "genre": ["Action", "Sci-Fi"],
            "rating": 8.7,
            "reason": f"Recommandé pour {current_user.username} basé sur ses préférences Sci-Fi"
        },
        {
            "id": 2,
            "title": "Inception",
            "year": 2010,
            "genre": ["Action", "Sci-Fi", "Thriller"],
            "rating": 8.8,
            "reason": f"Film complexe parfait pour {current_user.first_name}"
        }
    ]
    
    return {
        "user": current_user.username,
        "recommendations": mock_recommendations,
        "total": len(mock_recommendations)
    }

@router.get("/{username}", tags=["users"])
async def read_user(username: str):
    return {"username": username}