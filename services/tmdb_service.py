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

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MovieSearchFilters(BaseModel):
    genres: List[str] = Field(description="Liste de genres en français, ex: ['comédie', 'drame']")
    keywords: List[str] = Field(description="Liste de mots-clés significatifs en anglais")
    cast: List[str] = Field(description="Acteurs du film, ex: ['Tom Hanks', 'Liam Neeson']")
    directors: List[str] = Field(description="Réalisateurs du film, ex: ['Steven Spielberg']")


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

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Obtenir un film par son ID"""
        movies = self.load_movies()
        for movie in movies:
            if movie.get('id') == movie_id:
                return movie
        return None
    
    def get_similar_movies(self, base_movie_id: int, count: int = 10) -> List[Dict[str, Any]]:
        """Obtenir des films similaires basés sur les genres et mots-clés"""
        base_movie = self.get_movie_by_id(base_movie_id)
        if not base_movie:
            return self.get_random_movies(count)
        
        base_genres = base_movie.get('genres', [])
        base_keywords = base_movie.get('keywords', [])
        
        movies = self.load_movies()
        scored_movies = []
        
        for movie in movies:
            if movie.get('id') == base_movie_id:
                continue  # Ignorer le film de base
                
            score = 0
            movie_genres = movie.get('genres', [])
            movie_keywords = movie.get('keywords', [])
            
            # Score basé sur les genres communs
            common_genres = set(base_genres) & set(movie_genres)
            score += len(common_genres) * 2
            
            # Score basé sur les mots-clés communs
            common_keywords = set(base_keywords) & set(movie_keywords)
            score += len(common_keywords)
            
            if score > 0:
                scored_movies.append((movie, score))
        
        # Trier par score décroissant
        scored_movies.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les meilleurs films
        result = [movie for movie, score in scored_movies[:count]]
        
        # Si pas assez de films similaires, compléter avec des films aléaoires
        if len(result) < count:
            remaining = count - len(result)
            random_movies = self.get_random_movies(remaining * 2)  # Obtenir plus pour filtrer
            for movie in random_movies:
                if movie.get('id') not in [m.get('id') for m in result]:
                    result.append(movie)
                    if len(result) >= count:
                        break
        
        return result[:count]

    def search_movies_by_structured_filters(self, filters , count: int) -> List[Dict[str, Any]]:
        """Rechercher des films en fonction de filtres structurés venant d'un modèle LLM"""
        movies = self.load_movies()
        scored_movies = []
        logger.info(f"Searching for {filters}")
        for movie in movies:
            score = 0

            # Match genres
            movie_genres = [g.lower() for g in movie.get('genres', [])]
            for genre in filters.genres:
                if genre.lower() in movie_genres:
                    score += 3  # pondération importante

            # Match keywords
            movie_keywords = [k.lower() for k in movie.get('keywords', [])]
            for keyword in filters.keywords:
                if keyword.lower() in movie_keywords:
                    score += 2

            # Match cast
            movie_cast = [c.lower() for c in movie.get('cast', [])]
            for cast in filters.cast:
                if cast.lower() in movie_cast:
                    score += 2

            movie_directors = [c.lower() for c in movie.get('directors', [])]
            for title in filters.directors:
                if title.lower() in movie_directors:
                    score += 1

            if score > 0:
                scored_movies.append((movie, score))

        # Trier par score décroissant
        scored_movies.sort(key=lambda x: x[1], reverse=True)

        # Garder les meilleurs
        top_movies = [self._enhance_movie_data(m[0]) for m in scored_movies[:count]]

        # Compléter si nécessaire
        if len(top_movies) < count:
            top_movies += self.get_random_movies(count - len(top_movies))

        return top_movies[:count]

    def search_movies_by_prompt(self, prompt: str, count: int = 10) -> List[Dict[str, Any]]:
        """Rechercher des films basés sur un prompt utilisateur"""
        try:
            logger.info(f"User prompt: {prompt}")
            prompt_lower = prompt.lower()
            genre_keywords = self.parse_prompt_to_filters(prompt_lower)
            logger.debug(f"Parsed filters: {genre_keywords}")
            movies = self.search_movies_by_structured_filters(genre_keywords, count)
            logger.info(f"Found {len(movies)} movies from prompt.")
            return movies
        except Exception as e:
            logger.exception(f"Error in search_movies_by_prompt: {e}")
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

# Instance globale du service
tmdb_service = TMDBMovieService()
