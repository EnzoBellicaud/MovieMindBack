import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime

from models.Movie import Movie
from services.tmdb_service import tmdb_service
from services.vector_search import vector_search_service
from beanie import PydanticObjectId

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

# Models pour les requêtes
class ChatCreateRequest(BaseModel):
    prompt: str
    isGroupMode: bool = False

class ChatActionRequest(BaseModel):
    action: str  # "like", "dislike", "love"
    movie_id: int
    movie_title: str
    currentMovieIndex: int = Query(..., description="Index du film actuel dans la liste")

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
        prompt_lower = [request.prompt.lower()]
        filter = tmdb_service.parse_prompt_to_filters(prompt_lower)
        movie_count = 10
        movies = tmdb_service.search_movies_by_structured_filters(filter, movie_count)
        
        # Créer les données du chat
        chat_data = {
            "id": chat_id,
            "prompt": prompt_lower,
            "filter": filter,
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
async def chat_action(
        chat_id: str,
        request: ChatActionRequest
):
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

        print(len(chat_data["movies"]))

        # Si c'est un like, rafraîchir les films après l'index actuel
        if request.action == "like" and chat_data["user_preferences"]["liked"]:
            try:
                print("kldsmlksqmldkqlsmkdmlksqdmlksqmlkdmlksdmlksqml")
                # Récupérer les IDs des films likés depuis MongoDB
                liked_movie_ids = []
                for liked_movie in chat_data["user_preferences"]["liked"]:
                    movie = await Movie.find_one({"tmdb_id": liked_movie["id"]})
                    if movie:
                        liked_movie_ids.append(movie.id)

                if liked_movie_ids:
                    # Trouver le Movie correspondant au tmdb_id
                    liked_movie = await Movie.find_one({"tmdb_id": request.movie_id})
                    if liked_movie:
                        similar_results = await vector_search_service.find_similar_movies(
                            str(liked_movie.id),
                            limit=15
                        )

                    # Convertir les résultats en format TMDB pour le frontend
                    new_movies = []
                    existing_ids = {movie["id"] for movie in chat_data["movies"]}

                    for result in similar_results:
                        movie_data = result.movie
                        # Éviter les doublons
                        if movie_data.tmdb_id not in existing_ids:
                            # Convertir au format attendu par le frontend
                            tmdb_format = {
                                "id": movie_data.tmdb_id,
                                "title": movie_data.title,
                                "overview": movie_data.overview,
                                "release_date": movie_data.release_date,
                                "genres": movie_data.genres,
                                "vote_average": movie_data.vote_average,
                                "popularity": movie_data.popularity,
                                "poster_path": movie_data.poster_path,
                                "backdrop_path": movie_data.backdrop_path,
                                "poster_urls": {
                                    "w185": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w185{movie_data.poster_path}" if movie_data.poster_path else "",
                                    "w342": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w342{movie_data.poster_path}" if movie_data.poster_path else "",
                                    "w500": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w500{movie_data.poster_path}" if movie_data.poster_path else "",
                                    "w780": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w780{movie_data.poster_path}" if movie_data.poster_path else "",
                                    "original": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/original{movie_data.poster_path}" if movie_data.poster_path else ""
                                },
                                "backdrop_urls": {
                                    "w300": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w300{movie_data.backdrop_path}" if movie_data.backdrop_path else "",
                                    "w780": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w780{movie_data.backdrop_path}" if movie_data.backdrop_path else "",
                                    "w1280": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w1280{movie_data.backdrop_path}" if movie_data.backdrop_path else "",
                                    "original": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/original{movie_data.backdrop_path}" if movie_data.backdrop_path else ""
                                },
                                "poster_url": f"{tmdb_service.TMDB_IMAGE_BASE_URL}/w500{movie_data.poster_path}" if movie_data.poster_path else "",
                                "similarity_score": result.similarity_score
                            }
                            new_movies.append(tmdb_format)
                            existing_ids.add(movie_data.tmdb_id)

                    # Remplacer les films après l'index actuel
                    if new_movies and request.currentMovieIndex < len(chat_data["movies"]) - 1:
                        # Garder les films jusqu'à l'index actuel + 1
                        movies_to_keep = chat_data["movies"][:request.currentMovieIndex + 1]

                        # Ajouter les nouveaux films recommandés
                        # Limiter le nombre total de films
                        remaining_slots = 15 - len(movies_to_keep)
                        new_movies_to_add = new_movies[:remaining_slots]

                        chat_data["movies"] = movies_to_keep + new_movies_to_add
                        print(len(chat_data["movies"]))
                        logger.info(
                            f"Refreshed movies list after like. Kept {len(movies_to_keep)} movies, added {len(new_movies_to_add)} new recommendations")

            except Exception as e:
                logger.error(f"Error refreshing movies after like: {e}")
                # En cas d'erreur, on continue sans rafraîchir les films

        # Ajouter à l'historique
        action_text = {
            "like": f"✅ J'ai aimé \"{request.movie_title}\"",
            "dislike": f"❌ Je n'ai pas aimé \"{request.movie_title}\"",
            "love": f"💖 Coup de cœur pour \"{request.movie_title}\""
        }
        chat_data["conversation_history"].append(action_text[request.action])

    return {
        "success": True,
        "message": "Action enregistrée",
        "movies_updated": request.action == "like",
        "total_movies": len(chat_data["movies"]),
        "movies": chat_data["movies"],
    }

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
    refinement = request.refinement.lower()
    chat_data["conversation_history"].append(f"🔍 Affinement: \"{refinement}\"")
    chat_data["prompt"].append(refinement.lower())

    # Obtenir de nouveaux films basés sur l'affinement et les préférences
    refined_movies = []
    
    # Si l'utilisateur a des films aimés, chercher des films similaires
    if request.user_preferences.get("liked"):
        liked_movie_ids = [movie["id"] for movie in request.user_preferences["liked"]]
        for movie_id in liked_movie_ids[:2]:  # Prendre les 2 premiers films aimés
            similar_movies = tmdb_service.get_similar_movies(movie_id, count=5)
            refined_movies.extend(similar_movies)

    new_filter = tmdb_service.parse_prompt_to_filters(chat_data["prompt"])
    chat_data["filter"] = new_filter
    movie_count = 10
    additional_movies = tmdb_service.search_movies_by_structured_filters(new_filter, movie_count)
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
