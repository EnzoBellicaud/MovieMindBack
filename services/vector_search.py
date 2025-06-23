from typing import List, Dict, Any, Optional
import numpy as np
import logging
from models.Movie import Movie, VectorSearchResult, MovieResponse

logger = logging.getLogger(__name__)

class VectorSearchService:
    def __init__(self):
        self.sentence_model = None
        self.use_fallback = False
        self.movies_cache = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize models with error handling and fallback"""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence Transformer model loaded successfully")
            self.use_fallback = False
        except Exception as e:
            logger.warning(f"Sentence Transformer not available, using fallback: {e}")
            self.sentence_model = None
            self.use_fallback = True
            
    def _simple_text_vectorizer(self, text: str) -> List[float]:
        """Simple fallback vectorizer using basic text features"""
        if not text:
            return [0.0] * 100  # Return zero vector
            
        # Simple character-based features
        features = []
        text_lower = text.lower()
        
        # Length features
        features.append(len(text) / 1000.0)  # Normalized length
        features.append(len(text.split()) / 100.0)  # Normalized word count
        
        # Character frequency features (simplified)
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(text_lower.count(char) / len(text) if text else 0.0)
        
        # Common word features
        common_words = ['action', 'adventure', 'drama', 'comedy', 'thriller', 'horror', 
                       'romance', 'sci-fi', 'fantasy', 'crime', 'war', 'western',
                       'family', 'animation', 'documentary', 'musical', 'mystery',
                       'biography', 'history', 'sport', 'film', 'movie', 'story']
        
        for word in common_words:
            features.append(1.0 if word in text_lower else 0.0)
        
        # Pad or truncate to exactly 100 features
        while len(features) < 100:
            features.append(0.0)
        return features[:100]
            
    async def generate_movie_embeddings(self, movie: Movie) -> Dict[str, List[float]]:
        """Generate embeddings for a movie"""
        try:
            # Combine title and overview for better context
            title_text = movie.title or ""
            overview_text = movie.overview or ""
            genres_text = " ".join(movie.genres) if movie.genres else ""
            
            # Create combined text for embedding
            combined_text = f"{title_text} {overview_text} {genres_text}"
            
            if self.sentence_model and not self.use_fallback:
                # Use sentence transformer
                title_embedding = self.sentence_model.encode(title_text).tolist()
                overview_embedding = self.sentence_model.encode(overview_text).tolist() if overview_text else []
                combined_embedding = self.sentence_model.encode(combined_text).tolist()
            else:
                # Use fallback method
                title_embedding = self._simple_text_vectorizer(title_text)
                overview_embedding = self._simple_text_vectorizer(overview_text) if overview_text else []
                combined_embedding = self._simple_text_vectorizer(combined_text)
            
            return {
                "title_embedding": title_embedding,
                "overview_embedding": overview_embedding,
                "combined_embedding": combined_embedding
            }
        except Exception as e:
            logger.error(f"Error generating embeddings for movie {movie.title}: {e}")
            return {}
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
                
            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
        except Exception:
            return 0.0
    
    async def search_movies_by_text(self, query: str, limit: int = 20) -> List[VectorSearchResult]:
        """Search movies using vector similarity"""
        try:
            # Generate query embedding
            if self.sentence_model and not self.use_fallback:
                query_embedding = self.sentence_model.encode(query).tolist()
            else:
                query_embedding = self._simple_text_vectorizer(query)
            
            # Find movies with embeddings
            movies_with_embeddings = await Movie.find(
                Movie.combined_embedding.ne(None)
            ).to_list()
            
            results = []
            for movie in movies_with_embeddings:
                if movie.combined_embedding:
                    # Calculate similarity
                    similarity = self._calculate_similarity(query_embedding, movie.combined_embedding)
                    
                    results.append(VectorSearchResult(
                        movie=MovieResponse(
                            _id=str(movie.id),
                            tmdb_id=movie.tmdb_id,
                            title=movie.title,
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
                        ),
                        similarity_score=similarity
                    ))
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def find_similar_movies(self, movie_id: str, limit: int = 20) -> List[VectorSearchResult]:
        """Find movies similar to a given movie"""
        try:
            # Get the reference movie
            reference_movie = await Movie.get(movie_id)
            if not reference_movie or not reference_movie.combined_embedding:
                return []
            
            # Find other movies with embeddings
            movies_with_embeddings = await Movie.find(
                Movie.combined_embedding.ne(None),
                Movie.id != reference_movie.id
            ).to_list()
            
            results = []
            for movie in movies_with_embeddings:
                if movie.combined_embedding:
                    similarity = self._calculate_similarity(
                        reference_movie.combined_embedding, 
                        movie.combined_embedding
                    )
                    
                    results.append(VectorSearchResult(
                        movie=MovieResponse(
                            _id=str(movie.id),
                            tmdb_id=movie.tmdb_id,
                            title=movie.title,
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
                        ),
                        similarity_score=similarity
                    ))
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar movies: {e}")
            return []
    
    async def get_personalized_recommendations(self, user_id: str, limit: int = 20) -> List[VectorSearchResult]:
        """Get personalized movie recommendations for a user"""
        try:
            from models.User import User
            
            # Get user preferences
            user = await User.get(user_id)
            if not user:
                return []
            
            # Convert string IDs for query
            liked_movie_ids = [movie_id for movie_id in user.liked_movies if movie_id]
            disliked_movie_ids = [movie_id for movie_id in user.disliked_movies if movie_id]
            
            # Get user's liked movies
            liked_movies = []
            if liked_movie_ids:
                for movie_id in liked_movie_ids:
                    try:
                        movie = await Movie.get(movie_id)
                        if movie:
                            liked_movies.append(movie)
                    except Exception:
                        continue
            
            if not liked_movies:
                # If no liked movies, return popular movies
                return await self.get_popular_movies(limit)
            
            # Calculate average embedding of liked movies
            liked_embeddings = []
            for movie in liked_movies:
                if movie.combined_embedding:
                    liked_embeddings.append(movie.combined_embedding)
            
            if not liked_embeddings:
                return await self.get_popular_movies(limit)
            
            # Create user profile as average of liked movie embeddings
            user_profile = np.mean(liked_embeddings, axis=0).tolist()
            
            # Find movies similar to user profile (excluding liked and disliked)
            exclude_ids = liked_movie_ids + disliked_movie_ids
            movies_with_embeddings = await Movie.find(
                Movie.combined_embedding.ne(None)
            ).to_list()
            
            # Filter out movies user has already interacted with
            filtered_movies = [m for m in movies_with_embeddings if str(m.id) not in exclude_ids]
            
            results = []
            for movie in filtered_movies:
                if movie.combined_embedding:
                    similarity = self._calculate_similarity(user_profile, movie.combined_embedding)
                    
                    results.append(VectorSearchResult(
                        movie=MovieResponse(
                            _id=str(movie.id),
                            tmdb_id=movie.tmdb_id,
                            title=movie.title,
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
                        ),
                        similarity_score=similarity
                    ))
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return []
    
    async def get_popular_movies(self, limit: int = 20) -> List[VectorSearchResult]:
        """Get popular movies as fallback"""
        try:
            movies = await Movie.find().sort(-Movie.popularity).limit(limit).to_list()
            
            results = []
            for movie in movies:
                results.append(VectorSearchResult(
                    movie=MovieResponse(
                        _id=str(movie.id),
                        tmdb_id=movie.tmdb_id,
                        title=movie.title,
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
                    ),
                    similarity_score=movie.popularity / 100.0  # Normalize popularity as similarity
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting popular movies: {e}")
            return []

# Global instance
vector_search_service = VectorSearchService()
