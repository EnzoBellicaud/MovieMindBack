import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

# Imports pour Mistral API et embeddings
from langchain_mistralai import ChatMistralAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MistralMovieRecommendationSystem:
    def __init__(self,
                 movies_json_path: str,
                 mistral_api_key: str,
                 max_movies: int = 88797,
                 cache_dir: str = "./test_cache"):
        """
        Système de recommandation avec API Mistral pour tests (500 films)

        Args:
            movies_json_path: Chemin vers le fichier JSON
            mistral_api_key: Clé API Mistral
            max_movies: Nombre maximum de films à charger (500 par défaut)
            cache_dir: Répertoire de cache
        """
        self.movies_json_path = movies_json_path
        self.max_movies = max_movies
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialiser Mistral avec votre clé API
        self.llm = ChatMistralAI(
            model="mistral-medium-latest",
            api_key=mistral_api_key,
            temperature=0.7
        )

        # Charger les données (limité à max_movies)
        logger.info(f"Chargement des {max_movies} premiers films...")
        self.movies_data = self.load_limited_movies_data()
        logger.info(f"Chargé {len(self.movies_data)} films")

        # Initialiser les embeddings (modèle léger pour les tests)
        logger.info("Initialisation du modèle d'embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Créer la base vectorielle
        logger.info("Création de la base vectorielle...")
        self.vector_store, self.movie_metadata = self.create_vector_store()

        # Template pour les prompts Mistral
        self.explanation_template = PromptTemplate(
            input_variables=["user_preference", "movies_list"],
            template="""Tu es un expert en cinéma. Un utilisateur recherche des films basés sur cette préférence:
"{user_preference}"

Voici les films recommandés par le système:
{movies_list}

Explique en 2-3 phrases pourquoi ces films correspondent parfaitement à sa recherche. 
Sois enthousiaste et précis sur les points communs (genres, thèmes, ambiance, etc.).
Réponds en français."""
        )

    def load_limited_movies_data(self) -> List[Dict]:
        """Charge seulement les premiers films pour les tests"""
        with open(self.movies_json_path, 'r', encoding='utf-8') as f:
            all_movies = json.load(f)

        # Prendre seulement les premiers films
        limited_movies = all_movies[:self.max_movies]

        logger.info(f"Films chargés: {len(limited_movies)}/{len(all_movies)}")
        return limited_movies

    def create_movie_text(self, movie: Dict) -> str:
        """Crée un texte descriptif optimisé pour l'embedding"""
        if 'embedding_text' in movie and movie['embedding_text']:
            return movie['embedding_text']

        # Créer un texte concis mais informatif
        parts = []

        # Titre
        parts.append(f"Titre: {movie.get('title', '')}")

        # Titre original si différent
        if movie.get('original_title') != movie.get('title'):
            parts.append(f"Titre original: {movie.get('original_title', '')}")

        # Genres (très important)
        if movie.get('genres'):
            parts.append(f"Genres: {', '.join(movie['genres'])}")

        # Synopsis (tronqué si trop long)
        overview = movie.get('overview', '')
        if overview:
            overview = overview[:200] + "..." if len(overview) > 200 else overview
            parts.append(f"Synopsis: {overview}")

        # Mots-clés (excellent pour la similarité)
        if movie.get('keywords'):
            parts.append(f"Mots-clés: {', '.join(movie['keywords'][:8])}")

        # Acteurs principaux
        if movie.get('cast'):
            parts.append(f"Acteurs: {', '.join(movie['cast'][:3])}")

        # Réalisateur
        if movie.get('directors'):
            parts.append(f"Réalisateur: {', '.join(movie['directors'])}")

        # Année et note
        year = movie.get('release_date', '')[:4] if movie.get('release_date') else ''
        if year:
            parts.append(f"Année: {year}")

        if movie.get('vote_average'):
            parts.append(f"Note: {movie['vote_average']}/10")

        return "\n".join(parts)

    def create_vector_store(self) -> tuple:
        """Crée la base vectorielle avec les films limités"""
        documents = []
        movie_metadata = {}

        logger.info("Création des documents pour l'indexation...")
        for movie in tqdm(self.movies_data, desc="Préparation des films"):
            movie_text = self.create_movie_text(movie)

            metadata = {
                'id': movie['id'],
                'title': movie['title'],
                'genres': movie.get('genres', []),
                'vote_average': movie.get('vote_average', 0),
                'release_date': movie.get('release_date', ''),
                'popularity': movie.get('popularity', 0),
                'runtime': movie.get('runtime', 0),
                'tmdb_url': movie.get('tmdb_url', '')
            }

            doc = Document(page_content=movie_text, metadata=metadata)
            documents.append(doc)
            movie_metadata[movie['id']] = metadata

        # Créer la base vectorielle
        logger.info("Création de l'index FAISS...")
        vector_store = FAISS.from_documents(documents, self.embeddings)

        return vector_store, movie_metadata

    def find_similar_movies(self,
                            user_preference: str,
                            k: int = 5,
                            score_threshold: float = 0.6) -> List[Dict]:
        """Trouve des films similaires avec scores de similarité"""

        # Rechercher avec scores
        similar_docs_with_scores = self.vector_store.similarity_search_with_score(
            user_preference, k=k * 2  # Chercher plus pour filtrer
        )

        recommendations = []
        for doc, distance in similar_docs_with_scores:
            # Convertir distance en score de similarité (0-1)
            similarity_score = 1.0 / (1.0 + distance)

            # Filtrer par seuil
            if similarity_score < score_threshold:
                continue

            movie_info = {
                'title': doc.metadata['title'],
                'id': doc.metadata['id'],
                'genres': doc.metadata['genres'],
                'vote_average': doc.metadata['vote_average'],
                'release_date': doc.metadata['release_date'],
                'popularity': doc.metadata['popularity'],
                'runtime': doc.metadata['runtime'],
                'tmdb_url': doc.metadata.get('tmdb_url', ''),
                'similarity_score': round(similarity_score, 3)
            }

            recommendations.append(movie_info)

        # Trier par score de similarité
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)

        return recommendations[:k]

    def generate_explanation_with_mistral(self,
                                          user_preference: str,
                                          recommendations: List[Dict]) -> str:
        """Génère une explication avec l'API Mistral"""

        # Formater la liste des films
        movies_list = "\n".join([
            f"• {movie['title']} ({movie['release_date'][:4] if movie['release_date'] else 'N/A'}) - "
            f"Genres: {', '.join(movie['genres'])} - Note: {movie['vote_average']}/10 - "
            f"Similarité: {movie.get('similarity_score', movie.get('weighted_score', 'N/A'))}"
            for movie in recommendations
        ])

        # Créer le prompt
        prompt = self.explanation_template.format(
            user_preference=user_preference,
            movies_list=movies_list
        )

        try:
            # Appeler l'API Mistral
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"Erreur API Mistral: {e}")
            return f"Films recommandés basés sur vos préférences avec des scores de similarité élevés."

    def recommend_movies(self,
                         user_input: str,
                         k: int = 5,
                         include_explanation: bool = True,
                         filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fonction principale de recommandation

        Args:
            user_input: Préférence de l'utilisateur
            k: Nombre de recommandations
            include_explanation: Inclure l'explication Mistral
            filters: Filtres optionnels (genre, année, note...)
        """

        start_time = datetime.now()

        # Trouver des films similaires
        recommendations = self.find_similar_movies(user_input, k)

        # Appliquer des filtres si spécifiés
        if filters:
            recommendations = self._apply_filters(recommendations, filters)
            recommendations = recommendations[:k]  # Re-limiter après filtrage

        # Construire le résultat
        result = {
            'user_preference': user_input,
            'recommendations': recommendations,
            'total_movies_in_db': len(self.movies_data),
            'search_time_ms': round((datetime.now() - start_time).total_seconds() * 1000, 2),
            'filters_applied': filters or {}
        }

        # Générer l'explication avec Mistral si demandé
        if include_explanation and recommendations:
            try:
                explanation = self.generate_explanation_with_mistral(user_input, recommendations)
                result['explanation'] = explanation
            except Exception as e:
                logger.warning(f"Erreur explication Mistral: {e}")
                result['explanation'] = "Explication non disponible"

        return result

    def _apply_filters(self, movies: List[Dict], filters: Dict) -> List[Dict]:
        """Applique les filtres aux recommandations"""
        filtered = []

        for movie in movies:
            # Filtre par genre
            if 'genres' in filters:
                required_genres = set(filters['genres'])
                movie_genres = set(movie['genres'])
                if not required_genres.intersection(movie_genres):
                    continue

            # Filtre par note minimum
            if 'min_rating' in filters:
                if movie['vote_average'] < filters['min_rating']:
                    continue

            # Filtre par année
            if 'year_range' in filters:
                year_start, year_end = filters['year_range']
                movie_year = int(movie['release_date'][:4]) if movie['release_date'] else 0
                if not (year_start <= movie_year <= year_end):
                    continue

            # Filtre par durée
            if 'max_runtime' in filters:
                if movie['runtime'] > filters['max_runtime']:
                    continue

            filtered.append(movie)

        return filtered

    def recommend_from_liked_movies(self,
                                    liked_movies: List[Union[str, int]],
                                    k: int = 5,
                                    weights: Optional[List[float]] = None,
                                    include_explanation: bool = True) -> Dict[str, Any]:
        """
        Recommande des films basés sur plusieurs films aimés

        Args:
            liked_movies: Liste d'IDs (int) ou de titres (str) de films aimés
            k: Nombre de recommandations à retourner
            weights: Poids optionnels pour chaque film (même longueur que liked_movies)
            include_explanation: Inclure une explication Mistral

        Returns:
            Dict contenant les recommandations et métadonnées
        """
        start_time = datetime.now()

        # Trouver les films dans la base
        found_movies = []
        not_found = []

        for movie_ref in liked_movies:
            movie = self._find_movie_by_id_or_title(movie_ref)
            if movie:
                found_movies.append(movie)
            else:
                not_found.append(movie_ref)

        if not found_movies:
            return {
                'error': "Aucun des films spécifiés n'a été trouvé dans la base",
                'not_found': not_found
            }

        # Appliquer les poids si fournis
        if weights is None:
            weights = [1.0] * len(found_movies)
        elif len(weights) != len(liked_movies):
            weights = [1.0] * len(found_movies)
        else:
            # Ajuster les poids pour les films trouvés seulement
            weights = [w for i, w in enumerate(weights) if i < len(found_movies)]

        # Normaliser les poids
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Collecter tous les candidats avec leurs scores pondérés
        all_candidates = {}

        for movie, weight in zip(found_movies, weights):
            # Créer la description du film pour la recherche
            movie_description = self.create_movie_text(movie)

            # Rechercher des films similaires
            similar_docs = self.vector_store.similarity_search_with_score(
                movie_description,
                k=k * 3  # Chercher plus pour avoir assez de candidats
            )

            for doc, distance in similar_docs:
                movie_id = doc.metadata['id']

                # Exclure les films d'entrée
                if movie_id in [m['id'] for m in found_movies]:
                    continue

                # Calculer le score de similarité pondéré
                similarity_score = (1.0 / (1.0 + distance)) * weight

                # Accumuler les scores si le film apparaît plusieurs fois
                if movie_id in all_candidates:
                    all_candidates[movie_id]['weighted_score'] += similarity_score
                    all_candidates[movie_id]['matched_with'].append({
                        'title': movie['title'],
                        'weight': weight,
                        'similarity': 1.0 / (1.0 + distance)
                    })
                else:
                    all_candidates[movie_id] = {
                        'movie_info': {
                            'title': doc.metadata['title'],
                            'id': doc.metadata['id'],
                            'genres': doc.metadata['genres'],
                            'vote_average': doc.metadata['vote_average'],
                            'release_date': doc.metadata['release_date'],
                            'popularity': doc.metadata['popularity'],
                            'runtime': doc.metadata['runtime'],
                            'tmdb_url': doc.metadata.get('tmdb_url', '')
                        },
                        'weighted_score': similarity_score,
                        'matched_with': [{
                            'title': movie['title'],
                            'weight': weight,
                            'similarity': 1.0 / (1.0 + distance)
                        }]
                    }

        # Trier par score pondéré et prendre les k meilleurs
        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x['weighted_score'],
            reverse=True
        )[:k]

        # Formatter les recommandations
        recommendations = []
        for candidate in sorted_candidates:
            rec = candidate['movie_info'].copy()
            rec['weighted_score'] = round(candidate['weighted_score'], 3)
            rec['matched_with'] = candidate['matched_with']
            recommendations.append(rec)

        # Construire le résultat
        result = {
            'liked_movies': [
                {
                    'title': movie['title'],
                    'id': movie['id'],
                    'genres': movie.get('genres', []),
                    'weight': weight
                }
                for movie, weight in zip(found_movies, weights)
            ],
            'recommendations': recommendations,
            'not_found': not_found,
            'search_time_ms': round((datetime.now() - start_time).total_seconds() * 1000, 2)
        }

        # Générer l'explication si demandée
        if include_explanation and recommendations:
            liked_titles = [f"{m['title']} (poids: {w:.1f})"
                            for m, w in zip(found_movies, weights)]

            explanation_prompt = f"""Tu es un expert en cinéma. Un utilisateur a aimé ces films:
{', '.join(liked_titles)}

Voici les films recommandés par le système:
{self._format_recommendations_for_explanation(recommendations)}

Explique pourquoi ces recommandations sont pertinentes en te basant sur les films aimés.
Mentionne les points communs (genres, thèmes, style, époque, etc.).
Réponds en français en 3-4 phrases."""

            try:
                response = self.llm.invoke(explanation_prompt)
                result['explanation'] = response.content
            except Exception as e:
                logger.error(f"Erreur API Mistral: {e}")
                result['explanation'] = "Explication non disponible"

        return result

    def _find_movie_by_id_or_title(self, movie_ref: Union[str, int]) -> Optional[Dict]:
        """
        Trouve un film par son ID ou son titre

        Args:
            movie_ref: ID (int) ou titre partiel (str)

        Returns:
            Dict du film ou None si non trouvé
        """
        for movie in self.movies_data:
            # Recherche par ID
            if isinstance(movie_ref, int) and movie['id'] == movie_ref:
                return movie

            # Recherche par titre (insensible à la casse, recherche partielle)
            if isinstance(movie_ref, str):
                if movie_ref.lower() in movie['title'].lower():
                    return movie
                # Aussi vérifier le titre original
                if movie.get('original_title') and movie_ref.lower() in movie['original_title'].lower():
                    return movie

        return None

    def _format_recommendations_for_explanation(self, recommendations: List[Dict]) -> str:
        """Formate les recommandations pour le prompt d'explication"""
        lines = []
        for rec in recommendations[:5]:  # Limiter pour le prompt
            matched_films = [m['title'] for m in rec.get('matched_with', [])][:2]
            lines.append(
                f"• {rec['title']} ({rec['release_date'][:4] if rec['release_date'] else 'N/A'}) - "
                f"Genres: {', '.join(rec['genres'])} - Score: {rec['weighted_score']} - "
                f"Similaire à: {', '.join(matched_films)}"
            )
        return "\n".join(lines)

    def recommend_from_single_movie(self, movie_ref: Union[str, int], k: int = 5) -> Dict[str, Any]:
        """
        Version simplifiée pour un seul film (wrapper autour de recommend_from_liked_movies)

        Args:
            movie_ref: ID ou titre du film
            k: Nombre de recommandations

        Returns:
            Dict contenant les recommandations
        """
        return self.recommend_from_liked_movies([movie_ref], k=k)

    def get_database_stats(self) -> Dict[str, Any]:
        """Statistiques de la base de test"""

        genres_count = {}
        years = []
        ratings = []

        for movie in self.movies_data:
            # Genres
            for genre in movie.get('genres', []):
                genres_count[genre] = genres_count.get(genre, 0) + 1

            # Années
            if movie.get('release_date'):
                try:
                    year = int(movie['release_date'][:4])
                    years.append(year)
                except:
                    pass

            # Notes
            if movie.get('vote_average'):
                ratings.append(movie['vote_average'])

        return {
            'total_movies': len(self.movies_data),
            'top_genres': sorted(genres_count.items(), key=lambda x: x[1], reverse=True)[:10],
            'year_range': (min(years), max(years)) if years else None,
            'average_rating': round(np.mean(ratings), 2) if ratings else 0,
            'vector_dimensions': self.vector_store.index.d if hasattr(self.vector_store, 'index') else None
        }


# Exemple d'utilisation
if __name__ == "__main__":
    # Votre clé API Mistral
    MISTRAL_API_KEY = "BHP15ikr2vLZZL5FqNC1aSTEPd3qcPaR"

    # Initialiser le système (500 premiers films)
    print("Initialisation du système de recommandation...")
    recommender = MistralMovieRecommendationSystem(
        movies_json_path="tmdb_movies_for_embedding3.json",  # Votre fichier JSON
        mistral_api_key=MISTRAL_API_KEY,
        max_movies=88797
    )

    # Afficher les statistiques
    stats = recommender.get_database_stats()
    print("\n=== Statistiques de la base (88797 films) ===")
    print(f"Total films: {stats['total_movies']}")
    print(f"Années: {stats['year_range']}")
    print(f"Note moyenne: {stats['average_rating']}")
    print(f"Top 5 genres: {stats['top_genres'][:5]}")

    # Test 4: Recommandation basée sur plusieurs films aimés
    print("\n=== Test 4: Recommandation basée sur plusieurs films aimés ===")

    # Par titres
    liked_movies = ["The Matrix", "Inception", "Interstellar"]
    result4 = recommender.recommend_from_liked_movies(
        liked_movies=liked_movies,
        k=5,
        weights=[1.0, 0.8, 0.6]  # Plus de poids sur Matrix
    )

    if 'error' not in result4:
        print("Films aimés:")
        for movie in result4['liked_movies']:
            print(f"- {movie['title']} (poids: {movie['weight']:.1f})")

        if result4['not_found']:
            print(f"\nFilms non trouvés: {result4['not_found']}")

        print("\nRecommandations:")
        for i, rec in enumerate(result4['recommendations'], 1):
            print(f"\n{i}. {rec['title']} - Score pondéré: {rec['weighted_score']}")
            print(f"   Genres: {', '.join(rec['genres'])}")
            print(f"   Note: {rec['vote_average']}/10")
            print(f"   Similaire à:", end=" ")
            for match in rec['matched_with'][:2]:
                print(f"{match['title']} ({match['similarity']:.2f})", end=", ")
            print()

        if 'explanation' in result4:
            print(f"\nExplication:\n{result4['explanation']}")

    # Test 5: Recommandation par IDs
    print("\n=== Test 5: Recommandation par IDs de films ===")

    # Utilisons des IDs réels (vous devrez adapter selon votre base)
    # Pour l'exemple, cherchons d'abord les IDs
    matrix_id = None
    inception_id = None
    for movie in recommender.movies_data:
        if "matrix" in movie['title'].lower():
            matrix_id = movie['id']
        if "inception" in movie['title'].lower():
            inception_id = movie['id']

    if matrix_id and inception_id:
        movie_ids = [matrix_id, inception_id]
        result5 = recommender.recommend_from_liked_movies(
            liked_movies=movie_ids,
            k=3
        )

        print("Recommandations par IDs:")
        for i, rec in enumerate(result5['recommendations'], 1):
            print(f"{i}. {rec['title']} - Score: {rec['weighted_score']}")

    # Test 6: Mix IDs et titres
    print("\n=== Test 6: Mix IDs et titres ===")
    if matrix_id:
        mixed_refs = [matrix_id, "Inception", "The Dark Knight"]
        result6 = recommender.recommend_from_liked_movies(
            liked_movies=mixed_refs,
            k=4
        )

        if 'error' not in result6:
            print("Résultats avec mix IDs/titres:")
            for i, rec in enumerate(result6['recommendations'], 1):
                print(f"{i}. {rec['title']} - Score: {rec['weighted_score']}")

    # Test 7: Gestion des erreurs
    print("\n=== Test 7: Gestion des films non trouvés ===")
    result7 = recommender.recommend_from_liked_movies(
        liked_movies=["Film qui n'existe pas", "Autre film inexistant"],
        k=5
    )

    if 'error' in result7:
        print(f"Erreur: {result7['error']}")
        if 'not_found' in result7:
            print(f"Films non trouvés: {result7['not_found']}")