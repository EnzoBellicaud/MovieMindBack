import json
import random
from typing import List, Dict, Any, Optional

class TMDBMovieService:
    """Service pour gérer les films TMDB"""
    
    TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
    
    def __init__(self, json_file_path: str = "db/tmdb_movies_for_embedding3.json"):
        self.json_file_path = json_file_path
        self._movies_cache = None
    
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
                print(f"Loaded {len(self._movies_cache)} movies from TMDB")
            except Exception as e:
                print(f"Error loading movies: {e}")
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
    
    def search_movies_by_genre(self, genres: List[str], count: int = 10) -> List[Dict[str, Any]]:
        """Rechercher des films par genre avec données améliorées"""
        movies = self.load_movies()
        filtered_movies = []
        
        for movie in movies:
            movie_genres = movie.get('genres', [])
            if isinstance(movie_genres, list):
                # Vérifier si au moins un genre correspond
                if any(genre.lower() in [g.lower() for g in movie_genres] for genre in genres):
                    filtered_movies.append(movie)
        
        if len(filtered_movies) <= count:
            selected_movies = filtered_movies
        else:
            selected_movies = random.sample(filtered_movies, count)
        
        return [self._enhance_movie_data(movie) for movie in selected_movies]
    
    def search_movies_by_keywords(self, keywords: List[str], count: int = 10) -> List[Dict[str, Any]]:
        """Rechercher des films par mots-clés"""
        movies = self.load_movies()
        filtered_movies = []
        
        for movie in movies:
            movie_keywords = movie.get('keywords', [])
            movie_text = movie.get('embedding_text', '').lower()
            movie_overview = movie.get('overview', '').lower()
            
            # Rechercher dans les mots-clés, le texte d'embedding et l'overview
            match_found = False
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if (keyword_lower in movie_keywords or 
                    keyword_lower in movie_text or 
                    keyword_lower in movie_overview):
                    match_found = True
                    break
            
            if match_found:
                filtered_movies.append(movie)
        
        if len(filtered_movies) <= count:
            return filtered_movies
        return random.sample(filtered_movies, count)
    
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
    
    def search_movies_by_prompt(self, prompt: str, count: int = 10) -> List[Dict[str, Any]]:
        """Rechercher des films basés sur un prompt utilisateur"""
        prompt_lower = prompt.lower()
        
        # Mots-clés de genres communs
        genre_keywords = {
            'action': ['action', 'combat', 'baston', 'bagarre'],
            'comédie': ['drôle', 'rire', 'marrant', 'comique', 'humour'],
            'drame': ['émouvant', 'triste', 'drama', 'dramatique'],
            'horreur': ['peur', 'effrayant', 'horreur', 'épouvante'],
            'science-fiction': ['sci-fi', 'futur', 'espace', 'robot', 'technologie'],
            'romance': ['amour', 'romantique', 'couple'],
            'thriller': ['suspense', 'tension', 'mystère'],
            'aventure': ['aventure', 'exploration', 'voyage']
        }
        
        # Détecter les genres dans le prompt
        detected_genres = []
        for genre, keywords in genre_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_genres.append(genre)
        
        # Mots-clés spécifiques à rechercher
        keywords_to_search = []
        
        # Extraire des mots-clés du prompt
        words = prompt_lower.split()
        for word in words:
            if len(word) > 3:  # Ignorer les mots trop courts
                keywords_to_search.append(word)
        
        # Rechercher d'abord par genres si détectés
        if detected_genres:
            result = self.search_movies_by_genre(detected_genres, count)
            if result:
                return result
        
        # Puis par mots-clés
        if keywords_to_search:
            result = self.search_movies_by_keywords(keywords_to_search, count)
            if result:
                return result
        
        # En dernier recours, films aléaoires
        return self.get_random_movies(count)

# Instance globale du service
tmdb_service = TMDBMovieService()
