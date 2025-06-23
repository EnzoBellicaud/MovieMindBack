from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
import json
import logging
from models.Movie import (
    Movie, MovieResponse, MovieSearchRequest, MovieRecommendationRequest,
    MovieSimilarityRequest, MovieSwipeRequest, VectorSearchResult, MovieBulkInsert
)
from models.User import User
from services.vector_search import vector_search_service
from services.auth import get_current_user
from bson import ObjectId

router = APIRouter(prefix="/movies", tags=["movies"])
logger = logging.getLogger(__name__)

@router.post("/bulk-insert", response_model=Dict[str, Any])
async def bulk_insert_movies(bulk_data: MovieBulkInsert):
    """Bulk insert movies from TMDB data"""
    try:
        inserted_count = 0
        updated_count = 0
        
        for movie_data in bulk_data.movies:
            # Check if movie already exists
            existing_movie = await Movie.find_one(Movie.tmdb_id == movie_data.get("id"))
            
            if existing_movie:
                # Update existing movie
                for key, value in movie_data.items():
                    if key == "id":
                        continue
                    if key == "genre_ids":
                        setattr(existing_movie, key, value)
                    elif hasattr(existing_movie, key):
                        setattr(existing_movie, key, value)
                
                # Generate embeddings
                embeddings = await vector_search_service.generate_movie_embeddings(existing_movie)
                if embeddings:
                    existing_movie.title_embedding = embeddings.get("title_embedding")
                    existing_movie.overview_embedding = embeddings.get("overview_embedding")
                    existing_movie.combined_embedding = embeddings.get("combined_embedding")
                
                await existing_movie.replace()
                updated_count += 1
            else:
                # Create new movie
                new_movie = Movie(
                    tmdb_id=movie_data.get("id"),
                    title=movie_data.get("title", ""),
                    original_title=movie_data.get("original_title"),
                    overview=movie_data.get("overview"),
                    release_date=movie_data.get("release_date"),
                    genres=movie_data.get("genres", []),
                    genre_ids=movie_data.get("genre_ids", []),
                    adult=movie_data.get("adult", False),
                    original_language=movie_data.get("original_language", "en"),
                    popularity=movie_data.get("popularity", 0.0),
                    vote_average=movie_data.get("vote_average", 0.0),
                    vote_count=movie_data.get("vote_count", 0),
                    poster_path=movie_data.get("poster_path"),
                    backdrop_path=movie_data.get("backdrop_path"),
                    runtime=movie_data.get("runtime"),
                    budget=movie_data.get("budget"),
                    revenue=movie_data.get("revenue"),
                    production_companies=movie_data.get("production_companies", []),
                    production_countries=movie_data.get("production_countries", []),
                    spoken_languages=movie_data.get("spoken_languages", [])
                )
                
                # Generate embeddings
                embeddings = await vector_search_service.generate_movie_embeddings(new_movie)
                if embeddings:
                    new_movie.title_embedding = embeddings.get("title_embedding")
                    new_movie.overview_embedding = embeddings.get("overview_embedding")
                    new_movie.combined_embedding = embeddings.get("combined_embedding")
                
                await new_movie.insert()
                inserted_count += 1
        
        return {
            "message": "Bulk insert completed",
            "inserted": inserted_count,
            "updated": updated_count,
            "total_processed": len(bulk_data.movies)
        }
    
    except Exception as e:
        logger.error(f"Error in bulk insert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=List[VectorSearchResult])
async def search_movies(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100)
):
    """Search movies using vector similarity"""
    try:
        results = await vector_search_service.search_movies_by_text(query, limit)
        return results
    except Exception as e:
        logger.error(f"Error searching movies: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@router.get("/{movie_id}/similar", response_model=List[VectorSearchResult])
async def get_similar_movies(
    movie_id: str,
    limit: int = Query(20, ge=1, le=100)
):
    """Get movies similar to a specific movie"""
    try:
        results = await vector_search_service.find_similar_movies(movie_id, limit)
        return results
    except Exception as e:
        logger.error(f"Error finding similar movies: {e}")
        raise HTTPException(status_code=500, detail="Similar movies search failed")

@router.get("/recommendations", response_model=List[VectorSearchResult])
async def get_recommendations(
    current_user: User = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100)
):
    """Get personalized movie recommendations"""
    try:
        results = await vector_search_service.get_personalized_recommendations(
            str(current_user.id), limit
        )
        return results
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Recommendations failed")

@router.post("/swipe")
async def swipe_movie(
    swipe_data: MovieSwipeRequest,
    current_user: User = Depends(get_current_user)
):
    """Record user's swipe (like/dislike) on a movie"""
    try:
        user_id = ObjectId(swipe_data.user_id)
        movie_id = ObjectId(swipe_data.movie_id)
        
        # Verify the movie exists
        movie = await Movie.get(movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        # Update user preferences
        user = await User.get(current_user.id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if swipe_data.liked:
            if movie_id not in user.liked_movies:
                user.liked_movies.append(movie_id)
            if movie_id in user.disliked_movies:
                user.disliked_movies.remove(movie_id)
        else:
            if movie_id not in user.disliked_movies:
                user.disliked_movies.append(movie_id)
            if movie_id in user.liked_movies:
                user.liked_movies.remove(movie_id)
        
        await user.replace()
        
        return {
            "message": "Swipe recorded successfully",
            "liked": swipe_data.liked,
            "movie_title": movie.title
        }
    
    except Exception as e:
        logger.error(f"Error recording swipe: {e}")
        raise HTTPException(status_code=500, detail="Failed to record swipe")

@router.get("/popular", response_model=List[VectorSearchResult])
async def get_popular_movies(limit: int = Query(20, ge=1, le=100)):
    """Get popular movies"""
    try:
        results = await vector_search_service.get_popular_movies(limit)
        return results
    except Exception as e:
        logger.error(f"Error getting popular movies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get popular movies")

@router.get("/{movie_id}", response_model=MovieResponse)
async def get_movie(movie_id: str):
    """Get a specific movie by ID"""
    try:
        movie = await Movie.get(movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        return MovieResponse(
            _id=str(movie.id),
            tmdb_id=movie.tmdb_id,
            title=movie.title,
            original_title=movie.original_title,
            overview=movie.overview,
            release_date=movie.release_date,
            genres=movie.genres,
            poster_path=movie.poster_path,
            backdrop_path=movie.backdrop_path,
            adult=movie.adult,
            original_language=movie.original_language,
            popularity=movie.popularity,
            vote_average=movie.vote_average,
            vote_count=movie.vote_count,
            created_at=movie.created_at
        )
    except Exception as e:
        logger.error(f"Error getting movie: {e}")
        raise HTTPException(status_code=500, detail="Failed to get movie")

@router.get("/", response_model=List[MovieResponse])
async def get_movies(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    genre: Optional[str] = Query(None),
    min_rating: Optional[float] = Query(None, ge=0, le=10)
):
    """Get movies with optional filtering"""
    try:
        query_filter = {}
        
        if genre:
            query_filter[Movie.genres] = {"$in": [genre]}
        
        if min_rating:
            query_filter[Movie.vote_average] = {"$gte": min_rating}
        
        movies = await Movie.find(query_filter).skip(skip).limit(limit).to_list()
        
        return [
            MovieResponse(
                _id=str(movie.id),
                tmdb_id=movie.tmdb_id,
                title=movie.title,
                original_title=movie.original_title,
                overview=movie.overview,
                release_date=movie.release_date,
                genres=movie.genres,
                poster_path=movie.poster_path,
                backdrop_path=movie.backdrop_path,
                adult=movie.adult,
                original_language=movie.original_language,
                popularity=movie.popularity,
                vote_average=movie.vote_average,
                vote_count=movie.vote_count,
                created_at=movie.created_at
            ) for movie in movies
        ]
    except Exception as e:
        logger.error(f"Error getting movies: {e}")
        raise HTTPException(status_code=500, detail="Failed to get movies")