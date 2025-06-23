from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from beanie import Document
from datetime import datetime
import numpy as np

# Beanie Document model for Movies with vector embeddings
class Movie(Document):
    # TMDB data
    tmdb_id: int = Field(..., unique=True)
    title: str
    original_title: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    
    # Movie details
    genres: List[str] = []
    genre_ids: List[int] = []
    adult: bool = False
    original_language: str = "en"
    popularity: float = 0.0
    vote_average: float = 0.0
    vote_count: int = 0
    
    # Media
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    
    # Additional metadata
    runtime: Optional[int] = None
    budget: Optional[int] = None
    revenue: Optional[int] = None
    production_companies: List[Dict[str, Any]] = []
    production_countries: List[Dict[str, str]] = []
    spoken_languages: List[Dict[str, str]] = []
    
    # Vector embeddings for similarity search
    # We'll store embeddings as lists for MongoDB compatibility
    title_embedding: Optional[List[float]] = None
    overview_embedding: Optional[List[float]] = None
    combined_embedding: Optional[List[float]] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Recommendation metadata
    recommendation_score: float = 0.0
    similarity_features: Dict[str, Any] = {}
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class Settings:
        name = "movies"

# Pydantic models for API responses
class MovieBase(BaseModel):
    title: str
    original_title: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    genres: List[str] = []
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None

class MovieResponse(MovieBase):
    id: str = Field(alias="_id")
    tmdb_id: int
    adult: bool
    original_language: str
    popularity: float
    vote_average: float
    vote_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True
        populate_by_name = True

class MovieSearchRequest(BaseModel):
    query: str
    limit: int = 20
    genres: Optional[List[str]] = None
    min_rating: Optional[float] = None
    year_range: Optional[Dict[str, int]] = None

class MovieRecommendationRequest(BaseModel):
    user_id: str
    limit: int = 20
    exclude_seen: bool = True

class MovieSimilarityRequest(BaseModel):
    movie_id: str
    limit: int = 20

class MovieSwipeRequest(BaseModel):
    user_id: str
    movie_id: str
    liked: bool

class VectorSearchResult(BaseModel):
    movie: MovieResponse
    similarity_score: float
    
class MovieBulkInsert(BaseModel):
    movies: List[Dict[str, Any]]
