import json
import os
import random
from typing import List, Dict, Any, Optional
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema.runnable import Runnable
from pydantic import BaseModel, Field
import logging

from models.Movie import Movie
from services.vector_search import vector_search_service

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MovieSearchFilters(BaseModel):
    genres: List[str] = Field(default=[], description="Liste de genres en français, ex: ['comédie', 'drame']")
    keywords: List[str] = Field(default=[], description="Liste de mots-clés significatifs en anglais")
    cast: List[str] = Field(default=[], description="Acteurs du film, ex: ['Tom Hanks', 'Liam Neeson']")
    directors: List[str] = Field(default=[], description="Réalisateurs du film, ex: ['Steven Spielberg']")
    avg_embedding: List[float] = Field(default=[], description="Embedding moyen pour la recherche de similarité")


class TMDBMovieService:
    """Service pour gérer les films TMDB"""
    
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
    
    def __init__(self, json_file_path: str = "db/tmdb_movies_for_embedding3.json"):
        self.json_file_path = json_file_path
        self._movies_cache = None
        self._model = ChatMistralAI(
                model_name="mistral-medium-latest",
                api_key=os.getenv('MISTRAL_API_KEY')
            )
    
    def _enhance_movie_data(self, movie: Dict[str, Any]) -> Dict[str, Any]:
        """Améliorer les données d'un film avec des URLs d'images optimisées"""
        enhanced_movie = movie.copy()
        
        # S'assurer que les champs de poster sont présents
        poster_path = enhanced_movie.get('poster_path')
        backdrop_path = enhanced_movie.get('backdrop_path')
        
        # Ajouter des URLs d'images de différentes tailles
        enhanced_movie['poster_urls'] = {}
        enhanced_movie['backdrop_urls'] = {}
        
        if poster_path:
            enhanced_movie['poster_urls'] = {
                'w185': f"{self.TMDB_IMAGE_BASE_URL}/w185{poster_path}",
                'w342': f"{self.TMDB_IMAGE_BASE_URL}/w342{poster_path}",
                'w500': f"{self.TMDB_IMAGE_BASE_URL}/w500{poster_path}",
                'w780': f"{self.TMDB_IMAGE_BASE_URL}/w780{poster_path}",
                'original': f"{self.TMDB_IMAGE_BASE_URL}/original{poster_path}"
            }
        
        if backdrop_path:
            enhanced_movie['backdrop_urls'] = {
                'w300': f"{self.TMDB_IMAGE_BASE_URL}/w300{backdrop_path}",
                'w780': f"{self.TMDB_IMAGE_BASE_URL}/w780{backdrop_path}",
                'w1280': f"{self.TMDB_IMAGE_BASE_URL}/w1280{backdrop_path}",
                'original': f"{self.TMDB_IMAGE_BASE_URL}/original{backdrop_path}"
            }
        
        # Ajouter l'URL de poster par défaut pour compatibilité
        enhanced_movie['poster_url'] = enhanced_movie['poster_urls'].get('w500', '') if poster_path else ''
        
        return enhanced_movie

    def load_movies(self) -> List[Dict[str, Any]]:
        """Charger tous les films depuis le fichier JSON"""
        if self._movies_cache is None:
            try:
                with open(self.json_file_path, 'r', encoding='utf-8') as f:
                    self._movies_cache = json.load(f)
                logger.info(f"Loaded {len(self._movies_cache)} movies from TMDB")
            except FileNotFoundError as e:
                logger.error(f"File not found: {self.json_file_path}")
                self._movies_cache = []
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                self._movies_cache = []
            except Exception as e:
                logger.exception(f"Unexpected error while loading movies: {e}")
                self._movies_cache = []
        return self._movies_cache
    
    def get_random_movies(self, count: int = 10) -> List[Dict[str, Any]]:
        """Obtenir des films aléaoires avec données améliorées"""
        movies = self.load_movies()
        if len(movies) <= count:
            selected_movies = movies
        else:
            selected_movies = random.sample(movies, count)
        
        return [self._enhance_movie_data(movie) for movie in selected_movies]

    async def search_movies_by_structured_filters(self, filters, count: int) -> List[Dict[str, Any]]:
        """Rechercher des films en fonction de filtres structurés venant d'un modèle LLM"""
        movies = self.load_movies()
        scored_movies = []
        logger.info(f"Searching for {filters}")
        logger.info(f"Total movies to process: {len(movies)}")
        
        try:
            # Normaliser les filtres pour gérer à la fois les dict et les MovieSearchFilters
            if isinstance(filters, MovieSearchFilters):
                filters_dict = filters.dict()
            else:
                filters_dict = filters
            
            # Limiter le nombre de films à traiter pour éviter les boucles infinies
            movies_to_process = movies  # Limiter à 5000 films max
            processed_count = 0
            
            # Précalculer les embeddings si nécessaire
            has_embedding = len(filters_dict.get("avg_embedding", [])) > 0
            if has_embedding:
                logger.info("Recherche avec embedding activée")
            
            for movie in movies_to_process:
                processed_count += 1
                if processed_count % 1000 == 0:  # Log tous les 1000 films
                    logger.info(f"Processed {processed_count}/{len(movies_to_process)} movies")
                
                score = 0
                
                # Calcul de similarité (seulement si nécessaire et avec moins de logs)
                if has_embedding:
                    try:
                        movie_to_compare = await Movie.find_one({"tmdb_id": int(movie.get("id"))})
                        if movie_to_compare and hasattr(movie_to_compare, 'combined_embedding'):
                            similarity = vector_search_service.calculate_similarity(filters_dict["avg_embedding"], movie_to_compare.combined_embedding)
                            score += similarity
                    except Exception as e:
                        # Ne pas logger chaque erreur individuelle pour éviter le spam
                        pass
                
                # Match genres (plus efficace)
                movie_genres = set(g.lower() for g in movie.get('genres', []))
                for genre in filters_dict.get("genres", []):
                    if genre.lower() in movie_genres:
                        score += 3

                # Match keywords (plus efficace)
                movie_keywords = set(k.lower() for k in movie.get('keywords', []))
                for keyword in filters_dict.get("keywords", []):
                    if keyword.lower() in movie_keywords:
                        score += 2

                # Match cast (plus efficace)
                movie_cast = set(c.lower() for c in movie.get('cast', []))
                for cast in filters_dict.get("cast", []):
                    if cast.lower() in movie_cast:
                        score += 2

                # Match directors (plus efficace)
                movie_directors = set(d.lower() for d in movie.get('directors', []))
                for director in filters_dict.get("directors", []):
                    if director.lower() in movie_directors:
                        score += 1

                if score > 0:
                    scored_movies.append((movie, score))

            logger.info(f"Finished processing {processed_count} movies, found {len(scored_movies)} with scores")

            # Trier par score décroissant
            scored_movies.sort(key=lambda x: x[1], reverse=True)
            
            # Garder les meilleurs
            top_movies = [self._enhance_movie_data(m[0]) for m in scored_movies[:count]]
            
            # Compléter avec des films aléatoires si nécessaire
            if len(top_movies) < count:
                needed = count - len(top_movies)
                random_movies = self.get_random_movies(needed)
                top_movies.extend(random_movies)
            
            logger.info(f"Returning {len(top_movies)} movies")
            return top_movies[:count]
        
        except Exception as e:
            logger.exception(f"Error searching movies by structured filters: {e}")
            # En cas d'erreur, retourner des films aléatoires
            return self.get_random_movies(count)

    def parse_prompt_to_filters(self, user_prompt: List[str]) -> Dict[str, Any]:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Tu es un assistant qui transforme une description libre d'envie de film en critères de recherche structurés.")
            ]+ [
                ("user", prompt) for prompt in user_prompt
            ])

            chain = prompt | self._model.with_structured_output(schema=MovieSearchFilters)
            result = chain.invoke({})
            logger.debug(f"LLM structured result: {result}")
            return result
        except Exception as e:
            logger.exception(f"Error parsing user prompt to filters: {e}")
            return {}

    def get_similar_movies(self, movie_id: int, count: int = 5) -> List[Dict[str, Any]]:
        """Obtenir des films similaires à un film donné par ID"""
        movies = self.load_movies()
        # Pour l'instant, retourne des films aléaoires
        # TODO: Implémenter une vraie logique de similarité
        return self.get_random_movies(count)

# Instance globale du service
tmdb_service = TMDBMovieService()
