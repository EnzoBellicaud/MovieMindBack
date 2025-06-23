from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
from services.tmdb_service import tmdb_service

router = APIRouter(prefix="/chat", tags=["Chat"])

# Models pour les requêtes
class ChatCreateRequest(BaseModel):
    prompt: str
    isGroupMode: bool = False

class ChatActionRequest(BaseModel):
    action: str  # "like", "dislike", "love"
    movie_id: int
    movie_title: str

class ChatSelectRequest(BaseModel):
    movie_id: int
    movie_title: str

class ChatRefineRequest(BaseModel):
    refinement: str
    user_preferences: dict

# Stockage temporaire des chats (en production, utiliser une base de données)
CHAT_STORAGE = {}

@router.post("/", response_model=dict)
async def create_chat(request: ChatCreateRequest):
    """
    Créer un nouveau chat avec un prompt initial
    """
    try:        # Générer un ID unique pour le chat
        chat_id = str(uuid.uuid4())
        
        # Obtenir des films recommandés basés sur le prompt
        movies = tmdb_service.search_movies_by_prompt(request.prompt, count=15)
        
        # Créer les données du chat
        chat_data = {
            "id": chat_id,
            "prompt": request.prompt,
            "isGroupMode": request.isGroupMode,
            "movies": movies,
            "conversation_history": [f"Recherche initiale: \"{request.prompt}\""],
            "created_at": datetime.now().isoformat(),
            "user_preferences": {
                "liked": [],
                "disliked": [],
                "loved": []
            }
        }
        
        # Stocker le chat
        CHAT_STORAGE[chat_id] = chat_data
        
        return {"id": chat_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création du chat: {str(e)}")

@router.get("/{chat_id}", response_model=dict)
async def get_chat(chat_id: str):
    """
    Récupérer les données d'un chat par son ID
    """
    if chat_id not in CHAT_STORAGE:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    
    return CHAT_STORAGE[chat_id]

@router.post("/{chat_id}/action", response_model=dict)
async def chat_action(chat_id: str, request: ChatActionRequest):
    """
    Enregistrer une action utilisateur (like, dislike, love) sur un film
    """
    if chat_id not in CHAT_STORAGE:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    
    chat_data = CHAT_STORAGE[chat_id]
    
    # Ajouter l'action aux préférences utilisateur
    action_map = {
        "like": "liked",
        "dislike": "disliked", 
        "love": "loved"
    }
    
    if request.action in action_map:
        preference_type = action_map[request.action]
        movie_info = {
            "id": request.movie_id,
            "title": request.movie_title
        }
        chat_data["user_preferences"][preference_type].append(movie_info)
        
        # Ajouter à l'historique
        action_text = {
            "like": f"✅ J'ai aimé \"{request.movie_title}\"",
            "dislike": f"❌ Je n'ai pas aimé \"{request.movie_title}\"",
            "love": f"💖 Coup de cœur pour \"{request.movie_title}\""
        }
        chat_data["conversation_history"].append(action_text[request.action])
    
    return {"success": True, "message": "Action enregistrée"}

@router.post("/{chat_id}/select", response_model=dict)
async def chat_select(chat_id: str, request: ChatSelectRequest):
    """
    Enregistrer la sélection finale d'un film
    """
    if chat_id not in CHAT_STORAGE:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    
    chat_data = CHAT_STORAGE[chat_id]
    
    # Marquer le film comme sélectionné
    chat_data["selected_movie"] = {
        "id": request.movie_id,
        "title": request.movie_title,
        "selected_at": datetime.now().isoformat()
    }
    
    # Ajouter à l'historique
    chat_data["conversation_history"].append(f"🎯 Film sélectionné: \"{request.movie_title}\"")
    
    return {
        "success": True, 
        "message": "Film sélectionné",
        "movie": chat_data["selected_movie"]
    }

@router.post("/{chat_id}/refine", response_model=dict)
async def chat_refine(chat_id: str, request: ChatRefineRequest):
    """
    Affiner la recherche avec de nouveaux critères
    """
    if chat_id not in CHAT_STORAGE:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    
    chat_data = CHAT_STORAGE[chat_id]
      # Ajouter l'affinement à l'historique
    chat_data["conversation_history"].append(f"🔍 Affinement: \"{request.refinement}\"")
    
    # Obtenir de nouveaux films basés sur l'affinement et les préférences
    refined_movies = []
    
    # Si l'utilisateur a des films aimés, chercher des films similaires
    if request.user_preferences.get("liked"):
        liked_movie_ids = [movie["id"] for movie in request.user_preferences["liked"]]
        for movie_id in liked_movie_ids[:2]:  # Prendre les 2 premiers films aimés
            similar_movies = tmdb_service.get_similar_movies(movie_id, count=5)
            refined_movies.extend(similar_movies)
    
    # Compléter avec une recherche basée sur l'affinement
    additional_movies = tmdb_service.search_movies_by_prompt(request.refinement, count=10)
    refined_movies.extend(additional_movies)
    
    # Supprimer les doublons et limiter
    seen_ids = set()
    unique_movies = []
    for movie in refined_movies:
        if movie["id"] not in seen_ids:
            seen_ids.add(movie["id"])
            unique_movies.append(movie)
            if len(unique_movies) >= 15:
                break
    
    # Si pas assez de films, compléter avec des films aléaoires
    if len(unique_movies) < 10:
        random_movies = tmdb_service.get_random_movies(15 - len(unique_movies))
        for movie in random_movies:
            if movie["id"] not in seen_ids:
                unique_movies.append(movie)
    
    return {"movies": unique_movies[:15]}

@router.get("/{chat_id}/history", response_model=dict)
async def get_chat_history(chat_id: str):
    """
    Récupérer l'historique complet d'un chat
    """
    if chat_id not in CHAT_STORAGE:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    
    chat_data = CHAT_STORAGE[chat_id]
    
    return {
        "chat_id": chat_id,
        "conversation_history": chat_data["conversation_history"],
        "user_preferences": chat_data["user_preferences"],
        "selected_movie": chat_data.get("selected_movie")
    }

@router.delete("/{chat_id}", response_model=dict)
async def delete_chat(chat_id: str):
    """
    Supprimer un chat
    """
    if chat_id not in CHAT_STORAGE:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    
    del CHAT_STORAGE[chat_id]
    
    return {"success": True, "message": "Chat supprimé"}
