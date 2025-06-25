from typing import List, Dict, Any, Optional
import numpy as np
import logging
from models.Movie import Movie, VectorSearchResult, MovieResponse
from datetime import datetime

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
            # Utiliser un modèle multilingue pour mieux gérer les films français
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Multilingual Sentence Transformer model loaded successfully")
            self.use_fallback = False
        except Exception as e:
            logger.warning(f"Sentence Transformer not available, using fallback: {e}")
            self.sentence_model = None
            self.use_fallback = True

    def _create_rich_text_representation(self, movie: Movie) -> str:
        """Create a rich text representation of the movie using all available metadata"""
        parts = []

        # Titre et titre original
        if movie.title:
            parts.append(f"Titre: {movie.title}")
        if movie.original_title and movie.original_title != movie.title:
            parts.append(f"Titre original: {movie.original_title}")

        # Synopsis et tagline
        if movie.overview:
            parts.append(f"Synopsis: {movie.overview}")
        if hasattr(movie, 'tagline') and movie.tagline:
            parts.append(f"Slogan: {movie.tagline}")

        # Genres
        if movie.genres:
            parts.append(f"Genres: {', '.join(movie.genres)}")

        # Réalisateurs
        if hasattr(movie, 'directors') and movie.directors:
            parts.append(f"Réalisateurs: {', '.join(movie.directors)}")

        # Acteurs principaux
        if hasattr(movie, 'cast') and movie.cast:
            # Limiter aux 5 premiers acteurs
            main_cast = movie.cast[:5] if len(movie.cast) > 5 else movie.cast
            parts.append(f"Acteurs principaux: {', '.join(main_cast)}")

        # Mots-clés
        if hasattr(movie, 'keywords') and movie.keywords:
            parts.append(f"Mots-clés: {', '.join(movie.keywords)}")

        # Année de sortie
        if movie.release_date:
            try:
                year = movie.release_date.split('-')[0]
                parts.append(f"Année: {year}")
            except:
                pass

        # Note et popularité
        if movie.vote_average and movie.vote_average > 0:
            parts.append(f"Note: {movie.vote_average}/10")
        if movie.popularity and movie.popularity > 0:
            parts.append(f"Popularité: {movie.popularity}")

        # Budget et revenus (si disponibles et significatifs)
        if hasattr(movie, 'budget') and movie.budget and movie.budget > 1000000:
            parts.append(f"Budget: ${movie.budget:,}")
        if hasattr(movie, 'revenue') and movie.revenue and movie.revenue > 1000000:
            parts.append(f"Revenus: ${movie.revenue:,}")

        return "\n".join(parts)

    def _create_semantic_features(self, movie: Movie) -> Dict[str, float]:
        """Create semantic features from movie metadata"""
        features = {}

        # Popularité et note normalisées
        features['popularity_score'] = min(movie.popularity / 100.0, 1.0) if movie.popularity else 0.0
        features['rating_score'] = movie.vote_average / 10.0 if movie.vote_average else 0.0
        features['vote_confidence'] = min(movie.vote_count / 10000.0, 1.0) if movie.vote_count else 0.0

        # Âge du film (0 = très récent, 1 = très ancien)
        if movie.release_date:
            try:
                release_year = int(movie.release_date.split('-')[0])
                current_year = datetime.now().year
                age = current_year - release_year
                features['recency_score'] = max(0, 1 - (age / 50.0))  # Films de moins de 50 ans
            except:
                features['recency_score'] = 0.5
        else:
            features['recency_score'] = 0.5

        # Success commercial (ratio revenus/budget)
        if hasattr(movie, 'budget') and hasattr(movie, 'revenue'):
            if movie.budget and movie.budget > 0:
                roi = movie.revenue / movie.budget if movie.revenue else 0
                features['commercial_success'] = min(roi / 10.0, 1.0)  # ROI normalisé
            else:
                features['commercial_success'] = 0.5
        else:
            features['commercial_success'] = 0.5

        # Langue (bonus pour films en français)
        features['french_language'] = 1.0 if movie.original_language == 'fr' else 0.0
        features['english_language'] = 1.0 if movie.original_language == 'en' else 0.0

        # Genre features (genres populaires)
        genre_weights = {
            'Action': 0.9, 'Aventure': 0.85, 'Comédie': 0.95, 'Drame': 0.8,
            'Science-Fiction': 0.85, 'Thriller': 0.9, 'Horreur': 0.7,
            'Romance': 0.75, 'Animation': 0.8, 'Fantastique': 0.85,
            'Crime': 0.8, 'Mystère': 0.75, 'Documentaire': 0.6
        }

        if movie.genres:
            genre_scores = [genre_weights.get(genre, 0.7) for genre in movie.genres]
            features['genre_appeal'] = max(genre_scores) if genre_scores else 0.7
        else:
            features['genre_appeal'] = 0.7

        return features

    def _combine_embeddings(self, text_embedding: List[float], features: Dict[str, float],
                            weight_text: float = 0.85, weight_features: float = 0.15) -> List[float]:
        """Combine text embeddings with metadata features"""
        # Convertir les features en vecteur
        feature_vector = list(features.values())

        # Normaliser le vecteur de features pour qu'il ait la même dimension que text_embedding
        if len(feature_vector) < len(text_embedding):
            # Répéter les features pour atteindre la bonne dimension
            feature_vector = feature_vector * (len(text_embedding) // len(feature_vector) + 1)
            feature_vector = feature_vector[:len(text_embedding)]

        # Combiner les embeddings
        text_array = np.array(text_embedding)
        feature_array = np.array(feature_vector)

        # Normaliser les vecteurs
        text_norm = text_array / (np.linalg.norm(text_array) + 1e-8)
        feature_norm = feature_array / (np.linalg.norm(feature_array) + 1e-8)

        # Combinaison pondérée
        combined = weight_text * text_norm + weight_features * feature_norm

        # Re-normaliser le résultat
        combined_norm = combined / (np.linalg.norm(combined) + 1e-8)

        return combined_norm.tolist()

    async def generate_movie_embeddings(self, movie: Movie) -> Dict[str, List[float]]:
        """Generate enhanced embeddings for a movie using all available metadata"""
        try:
            # Créer la représentation textuelle enrichie
            rich_text = self._create_rich_text_representation(movie)

            # Créer les features sémantiques
            semantic_features = self._create_semantic_features(movie)

            # Textes individuels pour différents embeddings
            title_text = movie.title or ""
            overview_text = movie.overview or ""

            # Ajouter les mots-clés au titre pour un meilleur contexte
            if hasattr(movie, 'keywords') and movie.keywords:
                title_with_keywords = f"{title_text} - {', '.join(movie.keywords[:5])}"
            else:
                title_with_keywords = title_text

            if self.sentence_model and not self.use_fallback:
                # Générer les embeddings de base
                title_embedding = self.sentence_model.encode(title_with_keywords).tolist()
                overview_embedding = self.sentence_model.encode(overview_text).tolist() if overview_text else []
                rich_text_embedding = self.sentence_model.encode(rich_text).tolist()

                # Combiner avec les features pour l'embedding final
                combined_embedding = self._combine_embeddings(rich_text_embedding, semantic_features)
            else:
                # Utiliser la méthode de fallback améliorée
                title_embedding = self._enhanced_fallback_vectorizer(title_with_keywords, semantic_features)
                overview_embedding = self._enhanced_fallback_vectorizer(overview_text,
                                                                        semantic_features) if overview_text else []
                combined_embedding = self._enhanced_fallback_vectorizer(rich_text, semantic_features)

            return {
                "title_embedding": title_embedding,
                "overview_embedding": overview_embedding,
                "combined_embedding": combined_embedding
            }
        except Exception as e:
            logger.error(f"Error generating embeddings for movie {movie.title}: {e}")
            return {}

    def _enhanced_fallback_vectorizer(self, text: str, features: Dict[str, float]) -> List[float]:
        """Enhanced fallback vectorizer using text and metadata features"""
        if not text:
            return [0.0] * 100

        # Commencer avec le vectorizer de base
        base_vector = self._simple_text_vectorizer(text)

        # Ajouter les features metadata
        feature_values = list(features.values())

        # Combiner les vecteurs
        combined = base_vector[:90] + feature_values[:10]  # 90 features texte + 10 features metadata

        # S'assurer qu'on a exactement 100 features
        while len(combined) < 100:
            combined.append(0.0)

        return combined[:100]

    def _simple_text_vectorizer(self, text: str) -> List[float]:
        """Simple fallback vectorizer using basic text features"""
        if not text:
            return [0.0] * 100

        features = []
        text_lower = text.lower()

        # Length features
        features.append(len(text) / 1000.0)
        features.append(len(text.split()) / 100.0)

        # Character frequency features
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(text_lower.count(char) / len(text) if text else 0.0)

        # Genre keywords
        genre_keywords = {
            'action': ['action', 'combat', 'bataille', 'explosion', 'poursuite'],
            'comedy': ['comédie', 'drôle', 'rire', 'humour', 'comique'],
            'drama': ['drame', 'émotion', 'famille', 'vie', 'histoire'],
            'horror': ['horreur', 'peur', 'terrifiant', 'mort', 'sang'],
            'romance': ['amour', 'romance', 'couple', 'passion', 'coeur'],
            'scifi': ['science', 'fiction', 'futur', 'espace', 'robot'],
            'thriller': ['thriller', 'suspense', 'mystère', 'enquête', 'crime']
        }

        for genre, keywords in genre_keywords.items():
            score = sum(1.0 for keyword in keywords if keyword in text_lower) / len(keywords)
            features.append(score)

        # Common word features
        common_words = ['action', 'adventure', 'drama', 'comedy', 'thriller', 'horror',
                        'romance', 'sci-fi', 'fantasy', 'crime', 'war', 'western',
                        'family', 'animation', 'documentary', 'musical', 'mystery',
                        'biography', 'history', 'sport', 'film', 'movie', 'story']

        for word in common_words:
            features.append(1.0 if word in text_lower else 0.0)

        # Pad to 100 features
        while len(features) < 100:
            features.append(0.0)

        return features[:100]

    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0

            similarity = dot_product / (norm_v1 * norm_v2)
            return float(similarity)
        except Exception:
            return 0.0

    async def search_movies_by_text(self, query: str, limit: int = 20,
                                    boost_recent: bool = True,
                                    boost_popular: bool = True) -> List[VectorSearchResult]:
        """Enhanced search with metadata boosting"""
        try:
            # Générer l'embedding de la requête
            if self.sentence_model and not self.use_fallback:
                query_embedding = self.sentence_model.encode(query).tolist()
            else:
                # Pour le fallback, créer des features neutres
                neutral_features = {
                    'popularity_score': 0.5,
                    'rating_score': 0.5,
                    'vote_confidence': 0.5,
                    'recency_score': 0.5,
                    'commercial_success': 0.5,
                    'french_language': 0.0,
                    'english_language': 0.0,
                    'genre_appeal': 0.7
                }
                query_embedding = self._enhanced_fallback_vectorizer(query, neutral_features)

            # Récupérer les films avec embeddings
            movies_with_embeddings = await Movie.find(
                Movie.combined_embedding.ne(None)
            ).to_list()

            results = []
            for movie in movies_with_embeddings:
                if movie.combined_embedding:
                    # Calculer la similarité de base
                    base_similarity = self._calculate_similarity(query_embedding, movie.combined_embedding)

                    # Appliquer des boosts optionnels
                    final_score = base_similarity

                    if boost_recent and movie.release_date:
                        try:
                            year = int(movie.release_date.split('-')[0])
                            recency_boost = max(0, (year - 1990) / 35.0)  # Boost pour films après 1990
                            final_score *= (1 + 0.1 * recency_boost)
                        except:
                            pass

                    if boost_popular and movie.popularity:
                        popularity_boost = min(movie.popularity / 100.0, 1.0)
                        final_score *= (1 + 0.05 * popularity_boost)

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
                        similarity_score=final_score
                    ))

            # Trier par score final
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Error in enhanced vector search: {e}")
            return []

    async def find_similar_movies(self, movie_id: str, limit: int = 20) -> List[VectorSearchResult]:
        """Find movies similar to a given movie"""
        try:
            reference_movie = await Movie.get(movie_id)
            if not reference_movie or not reference_movie.combined_embedding:
                return []

            movies_with_embeddings = await Movie.find({
                "combined_embedding": {"$ne": None},
                "_id": {"$ne": reference_movie.id}
            }).to_list()

            results = []
            for movie in movies_with_embeddings:
                if movie.combined_embedding:
                    similarity = self._calculate_similarity(
                        reference_movie.combined_embedding,
                        movie.combined_embedding
                    )

                    # Bonus pour les films du même genre
                    if reference_movie.genres and movie.genres:
                        common_genres = set(reference_movie.genres) & set(movie.genres)
                        genre_bonus = len(common_genres) / max(len(reference_movie.genres), len(movie.genres))
                        similarity *= (1 + 0.2 * genre_bonus)

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

            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Error finding similar movies: {e}")
            return []

    async def recommend_movie_from_list(self, tmdb_ids: List[str], limit: int = 1) -> List[VectorSearchResult]:
        """Recommend a movie based on a list of reference movies"""
        try:
            # Récupérer tous les films de référence
            reference_movies = []
            for tmdb_id in tmdb_ids:
                movie = await Movie.find_one({"tmdb_id": int(tmdb_id)})
                if movie and movie.combined_embedding:
                    reference_movies.append(movie)

            if not reference_movies:
                return []

            # Calculer l'embedding moyen des films de référence
            avg_embedding = self._calculate_average_embedding([m.combined_embedding for m in reference_movies])

            # Récupérer tous les autres films avec embeddings
            all_movies = await Movie.find({
                "combined_embedding": {"$ne": None},
                "_id": {"$nin": [movie.id for movie in reference_movies]}
            }).to_list()

            results = []
            for movie in all_movies:
                if movie.combined_embedding:
                    # Calculer la similarité avec l'embedding moyen
                    similarity = self._calculate_similarity(avg_embedding, movie.combined_embedding)

                    # Bonus pour les genres communs avec les films de référence
                    genre_bonus = self._calculate_genre_bonus(reference_movies, movie)
                    similarity *= (1 + 0.2 * genre_bonus)

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

            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Error recommending movie from list: {e}")
            return []

    def _calculate_average_embedding(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate the average embedding from a list of embeddings"""
        if not embeddings:
            return []

        # Supposer que tous les embeddings ont la même dimension
        embedding_dim = len(embeddings[0])
        avg_embedding = [0.0] * embedding_dim

        for embedding in embeddings:
            for i, value in enumerate(embedding):
                avg_embedding[i] += value

        # Diviser par le nombre d'embeddings pour obtenir la moyenne
        for i in range(embedding_dim):
            avg_embedding[i] /= len(embeddings)

        return avg_embedding

    def _calculate_genre_bonus(self, reference_movies: List[Movie], candidate_movie: Movie) -> float:
        """Calculate genre bonus based on overlap with reference movies"""
        if not candidate_movie.genres:
            return 0.0

        # Collecter tous les genres des films de référence
        all_reference_genres = set()
        for ref_movie in reference_movies:
            if ref_movie.genres:
                all_reference_genres.update(ref_movie.genres)

        if not all_reference_genres:
            return 0.0

        # Calculer le chevauchement
        candidate_genres = set(candidate_movie.genres)
        common_genres = all_reference_genres & candidate_genres

        return len(common_genres) / len(all_reference_genres)

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