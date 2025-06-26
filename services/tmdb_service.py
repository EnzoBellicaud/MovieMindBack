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
import numpy as np
import heapq
import numpy as np
from typing import List, Dict, Any, Tuple

from models.Movie import Movie, MovieResponse
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
    vote_average: Optional[float] = Field(default=None, description="Note minimale du film (0-10)")
    release_date: Optional[str] = Field(default=None, description="Date de sortie du film au format 'YYYY-MM-DD'")
    runtime: Optional[int] = Field(default=None, description="Durée du film en minutes")
    
    avg_embedding: List[float] = Field(default=[], description="Embedding moyen pour la recherche de similarité")


class TMDBMovieService:
    """Service pour gérer les films TMDB"""
    
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
    
    def __init__(self, json_file_path: str = "db/tmdb_movies_for_embedding3.json"):
        self.json_file_path = json_file_path
        self._movies_cache = None
        self._model = ChatMistralAI(
                model_name="mistral-medium-latest",
                api_key=os.getenv('MISTRAL_API_KEY'),
                temperature=0.1,
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
        """
        Optimized search combining MongoDB pre-filtering, percentage-based similarity scoring,
        optional batched semantic similarity, and top-k heap selection.
        """
        # 1. Build and execute MongoDB query to pre-filter candidates
        try:
            query = _build_mongo_query(filters)
            raw_movies = await Movie.find(query).to_list()

            # 2. Compute percentage-based similarity scores (0-100)
            sim_scores = _compute_similarity_scores(raw_movies, filters)
            
            # Keep only those with non-zero similarity to limit work
            candidates = [(m, s) for m, s in sim_scores if s > 0]
            

            # 3. Compute batch semantic similarities if embedding provided
  
            
            if hasattr(filters, 'avg_embedding') and filters.avg_embedding and candidates:
                
                movie_ids = [int(m.get('id')) for m, _ in candidates]
                db_movies = await Movie.find({"tmdb_id": {"$in": movie_ids}}).to_list()
                emb_map = {m.tmdb_id: m.combined_embedding for m in db_movies if m.combined_embedding}
                sem_scores = _compute_batch_similarities(filters.avg_embedding, candidates, emb_map)
                # Combine scores: weighted sum of percentage and semantic similarity (also normalized 0-100)
                ALPHA = 0.6  # weight towards structured similarity
                total_scores = []
                for (movie, crit_pct), sem in zip(candidates, sem_scores):
                    sem_pct = sem * 100
                    combined = ALPHA * crit_pct + (1 - ALPHA) * sem_pct
                    total_scores.append((movie, combined))
            else:
                logger.info("No avg_embedding provided, using percentage scores only")
                total_scores = candidates
                

            # 4. Top-k selection via heap (highest combined percentage)
            top_k = heapq.nlargest(count, total_scores, key=lambda x: x[1])

            # 5. Enhance and return
        except Exception as e:
            logger.exception(f"Error during movie search: {e}")
            return []
        
        logger.info(f"top_k: {len(top_k)} movies selected")
        
        return [self._enhance_movie_data(movie.model_dump()) for movie, _ in top_k]


   

    def parse_prompt_to_filters(self, user_prompt: List[str]) -> Dict[str, Any]:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                "Tu es un expert du cinéma. Ton rôle est de transformer une description libre des envies de l'utilisateur "
                "en un ensemble **structuré et très complet** de filtres pour rechercher des films.\n"
                "Tu dois générer des listes **exhaustives** pour chaque critère pertinent :\n"
                "- `genres` : tous les genres associés (en français)\n"
                "- `keywords` : une grande variété de mots-clés significatifs en anglais (thèmes, lieux, ambiances, personnages...)\n"
                "- `cast` : noms des acteurs emblématiques associés au sujet\n"
                "- `directors` : réalisateurs liés à ce type de film\n\n"
                "Exemples :\n"
                "Si l'utilisateur dit « j’ai envie d’un film comme Spiderman », tu devras inclure :\n"
                "genres : ['action', 'super-héros', 'aventure']\n"
                "keywords : ['spider', 'hero', 'Marvel', 'web', 'villain', 'New York', 'responsibility', 'origin story']\n"
                "cast : ['Tobey Maguire', 'Andrew Garfield', 'Tom Holland', 'Zendaya']\n"
                "directors : ['Sam Raimi', 'Marc Webb', 'Jon Watts']\n\n"
                "Sois généreux et large dans les suggestions, même si le prompt est vague. "
                "Mieux vaut trop de filtres que pas assez."
                )
            ] + [
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


def _build_mongo_query(filters) -> Dict[str, Any]:
    """Construit une requête MongoDB où runtime et release_date sont requis, les autres en $or."""
    fd = filters.dict() if hasattr(filters, 'dict') else filters
    or_clauses = []
    if fd.get('genres'):
        or_clauses.append({'genres': {'$in': fd['genres']}})
    if fd.get('keywords'):
        or_clauses.append({'keywords': {'$in': fd['keywords']}})
    if fd.get('cast'):
        or_clauses.append({'cast': {'$in': fd['cast']}})
    if fd.get('directors'):
        or_clauses.append({'directors': {'$in': fd['directors']}})
    if fd.get('vote_average') is not None:
        or_clauses.append({'vote_average': {'$gte': fd['vote_average']}})

    # Clauses obligatoires
    must_conditions = []
    if fd.get('runtime') is not None:
        must_conditions.append({'runtime': {'$gte': fd['runtime']}})
    if fd.get('release_date'):
        must_conditions.append({'release_date': {'$gte': fd['release_date']}})

    # Si aucune condition obligatoire, on retourne juste le OR
    if not must_conditions:
        return {'$or': or_clauses} if or_clauses else {}

    # Si aucune condition optionnelle, on retourne juste le AND des obligatoires
    if not or_clauses:
        return {'$and': must_conditions}

    # Sinon, on combine les deux
    return {
        '$and': [
            {'$or': or_clauses},
            *must_conditions
        ]
    }



def _compute_similarity_scores(movies: List[Dict[str, Any]], filters) -> List[Tuple[Dict[str, Any], float]]:
    """Compute percentage similarity between movie attributes and filter terms."""
    fd = filters.dict() if hasattr(filters, 'dict') else filters
    # Extract filter lists
    f_genres = [g.lower() for g in fd.get('genres', [])]
    f_keywords = [k.lower() for k in fd.get('keywords', [])]
    f_cast = [c.lower() for c in fd.get('cast', [])]
    f_dirs = [d.lower() for d in fd.get('directors', [])]
    
    
    o_release_date = fd.get('release_date')
    o_runtime = fd.get('runtime')
    o_vote_average = fd.get('vote_average')

    # Total filter term counts
    total_terms = len(f_genres) + len(f_keywords) + len(f_cast) + len(f_dirs) + \
        (1 if o_release_date else 0) + \
        (1 if o_runtime else 0) + \
        (1 if o_vote_average is not None else 0)
    scores: List[Tuple[Dict[str, Any], float]] = []
    if total_terms == 0:
        return [(m, 0.0) for m in movies]
    


    for movie in movies:

        # Lowercase sets for comparison
        mg = set(g.lower() for g in (movie.genres or []))
        mk = set(k.lower() for k in (movie.keywords or []))
        mc = set(c.lower() for c in (movie.cast or []))
        md = set(d.lower() for d in (movie.directors or []))
        
        if movie.vote_average is not None:
            logger.info(f"movie.vote_average: {movie.vote_average}")
        # Count matches
        match_count = (
            sum(1 for g in f_genres if g in mg) +
            sum(1 for k in f_keywords if k in mk) +
            sum(1 for c in f_cast if c in mc) +
            sum(1 for d in f_dirs if d in md) +
            (1 if o_release_date and movie.release_date and movie.release_date >= o_release_date else 0) +
            (1 if o_runtime and movie.runtime and movie.runtime >= o_runtime else 0) +
            (1 if o_vote_average is not None and movie.vote_average is not None and movie.vote_average >= o_vote_average else 0)
        )
        
        
        # Percentage similarity 0-100
        pct = (match_count / total_terms) * 100
        
        scores.append((movie, pct))
        
 
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores


def _compute_batch_similarities(query_emb: List[float],
                            candidates: List[Tuple[Dict[str, Any], float]],
                            emb_map: Dict[int, List[float]]) -> List[float]:
    """Compute cosine similarities in batch for candidate movies with embeddings (0-1 range)."""
    embeddings = []
    for movie, _ in candidates:
        mid = int(movie.get('id'))
        if mid in emb_map:
            embeddings.append(emb_map[mid])
        else:
            embeddings.append([0.0] * len(query_emb))
    emb_matrix = np.array(embeddings)
    # Normalized
    qe = np.array(query_emb)
    qe_norm = qe / np.linalg.norm(qe)
    M = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    sims = M.dot(qe_norm)
    # Clip NaNs
    sims = np.nan_to_num(sims)
    return sims.tolist()
