#!/usr/bin/env python3
"""
Script de migration des films depuis tmdb_movies_for_embedding3.json vers MongoDB
avec génération d'embeddings vectoriels
"""

import asyncio
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from models.Movie import Movie

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_movies_to_mongodb():
    """Migrer les films depuis JSON vers MongoDB avec embeddings"""
    try:
        # Import des modules après configuration
        from db.init_db import init_db
        from models.Movie import Movie
        from services.vector_search import vector_search_service
        
        # Initialiser la base de données
        await init_db()
        logger.info("Base de données initialisée")
        
        # Charger les données des films
        json_file_path = Path("db/tmdb_movies_for_embedding3.json")
        
        if not json_file_path.exists():
            logger.error(f"Fichier {json_file_path} non trouvé")
            return
        
        logger.info(f"Chargement des données depuis {json_file_path}")
        
        # Lire le fichier JSON par chunks pour éviter les problèmes de mémoire
        batch_size = 100
        total_processed = 0
        total_inserted = 0
        total_updated = 0
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                movies_data = json.load(f)
                
            if isinstance(movies_data, list):
                movies_list = movies_data
            else:
                # Si c'est un dictionnaire, essayer de trouver la liste des films
                movies_list = movies_data.get('movies', movies_data.get('results', []))
            
            logger.info(f"Trouvé {len(movies_list)} films à traiter")
            
            # Traiter par batches
            for i in range(0, len(movies_list), batch_size):
                batch = movies_list[i:i + batch_size]
                batch_inserted, batch_updated = await process_movie_batch(batch, vector_search_service)
                
                total_processed += len(batch)
                total_inserted += batch_inserted
                total_updated += batch_updated
                
                logger.info(f"Traité {total_processed}/{len(movies_list)} films")
                
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON: {e}")
            return
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            return
        
        logger.info(f"""
        Migration terminée:
        - Total traité: {total_processed}
        - Nouveaux films insérés: {total_inserted}
        - Films mis à jour: {total_updated}
        """)
        
    except Exception as e:
        logger.error(f"Erreur lors de la migration: {e}")
        raise

async def process_movie_batch(batch: List[Dict[str, Any]], vector_service) -> tuple[int, int]:
    """Traiter un batch de films"""
    inserted_count = 0
    updated_count = 0
    
    for movie_data in batch:
        try:
            # Extraire les données du film
            tmdb_id = movie_data.get("id")
            if not tmdb_id:
                logger.warning("Film sans ID TMDB, ignoré")
                continue
            
            # Vérifier si le film existe déjà
            existing_movie = await Movie.find_one(Movie.tmdb_id == tmdb_id)
            
            # Préparer les données du film
            movie_dict = prepare_movie_data(movie_data)
            
            if existing_movie:
                # Mettre à jour le film existant
                for key, value in movie_dict.items():
                    if hasattr(existing_movie, key):
                        setattr(existing_movie, key, value)
                
                # Générer les embeddings
                embeddings = await vector_service.generate_movie_embeddings(existing_movie)
                if embeddings:
                    existing_movie.title_embedding = embeddings.get("title_embedding")
                    existing_movie.overview_embedding = embeddings.get("overview_embedding")
                    existing_movie.combined_embedding = embeddings.get("combined_embedding")
                
                await existing_movie.replace()
                updated_count += 1
                
            else:
                # Créer un nouveau film
                new_movie = Movie(**movie_dict)
                
                # Générer les embeddings
                embeddings = await vector_service.generate_movie_embeddings(new_movie)
                if embeddings:
                    new_movie.title_embedding = embeddings.get("title_embedding")
                    new_movie.overview_embedding = embeddings.get("overview_embedding")
                    new_movie.combined_embedding = embeddings.get("combined_embedding")
                
                await new_movie.insert()
                inserted_count += 1
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement du film {movie_data.get('id', 'unknown')}: {e}")
            continue
    
    return inserted_count, updated_count

def prepare_movie_data(movie_data: Dict[str, Any]) -> Dict[str, Any]:
    """Préparer les données du film pour MongoDB"""
    return {
        "tmdb_id": movie_data.get("id"),
        "title": movie_data.get("title", ""),
        "original_title": movie_data.get("original_title"),
        "overview": movie_data.get("overview"),
        "release_date": movie_data.get("release_date"),
        "genres": extract_genres(movie_data.get("genres", [])),
        "genre_ids": movie_data.get("genre_ids", []),
        "adult": movie_data.get("adult", False),
        "original_language": movie_data.get("original_language", "en"),
        "popularity": float(movie_data.get("popularity", 0.0)),
        "vote_average": float(movie_data.get("vote_average", 0.0)),
        "vote_count": int(movie_data.get("vote_count", 0)),
        "poster_path": movie_data.get("poster_path"),
        "backdrop_path": movie_data.get("backdrop_path"),
        "runtime": movie_data.get("runtime"),
        "budget": movie_data.get("budget"),
        "revenue": movie_data.get("revenue"),
        "production_companies": movie_data.get("production_companies", []),
        "production_countries": movie_data.get("production_countries", []),
        "spoken_languages": movie_data.get("spoken_languages", [])
    }

def extract_genres(genres_data) -> List[str]:
    """Extraire les noms des genres"""
    if isinstance(genres_data, list):
        return [genre.get("name", "") for genre in genres_data if isinstance(genre, dict)]
    return []

async def create_sample_data():
    """Créer des données d'exemple si le fichier JSON n'existe pas"""
    sample_movies = [
        {
            "id": 1,
            "title": "The Shawshank Redemption",
            "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
            "release_date": "1994-09-23",
            "genres": [{"name": "Drama"}],
            "genre_ids": [18],
            "adult": False,
            "original_language": "en",
            "popularity": 67.931,
            "vote_average": 8.7,
            "vote_count": 26000,
            "poster_path": "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg",
            "backdrop_path": "/iNh3BivHyg5sQRPP1KOkzguEX0H.jpg"
        },
        {
            "id": 2,
            "title": "The Godfather",
            "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
            "release_date": "1972-03-24",
            "genres": [{"name": "Crime"}, {"name": "Drama"}],
            "genre_ids": [80, 18],
            "adult": False,
            "original_language": "en",
            "popularity": 65.466,
            "vote_average": 8.7,
            "vote_count": 18000,
            "poster_path": "/3bhkrj58Vtu7enYsRolD1fZdja1.jpg",
            "backdrop_path": "/mDfJG3LC3Dqb67AZ52x3Z0jU0uB.jpg"
        }
    ]
    
    # Sauvegarder les données d'exemple
    with open("db/sample_movies.json", "w", encoding="utf-8") as f:
        json.dump(sample_movies, f, indent=2, ensure_ascii=False)
    
    logger.info("Données d'exemple créées dans db/sample_movies.json")
    return sample_movies

if __name__ == "__main__":
    asyncio.run(migrate_movies_to_mongodb())
