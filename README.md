# MovieMind Backend - API avec MongoDB et Recherche Vectorielle

## 🎬 Description

Backend FastAPI pour MovieMind avec base de données MongoDB et système de recherche vectorielle pour les films.

## 🛠 Technologies

- **FastAPI** - Framework web moderne et rapide
- **MongoDB** - Base de données NoSQL avec support vectoriel
- **Beanie** - ODM (Object Document Mapper) pour MongoDB
- **Sentence Transformers** - Génération d'embeddings vectoriels
- **Motor** - Driver MongoDB asynchrone
- **JWT** - Authentification par tokens

## 🏗 Architecture

```
MovieMindBack/
├── main.py                 # Point d'entrée de l'application
├── requirements.txt        # Dépendances Python
├── docker-compose.yml      # Configuration Docker
├── Dockerfile             # Image Docker de l'API
├── .env.example           # Variables d'environnement exemple
├── db/
│   ├── init_db.py         # Configuration MongoDB
│   └── tmdb_movies_for_embedding3.json  # Données films
├── models/
│   ├── User.py            # Modèle utilisateur MongoDB
│   └── Movie.py           # Modèle film avec embeddings
├── routes/
│   ├── auth.py            # Routes d'authentification
│   ├── movies.py          # Routes des films
│   ├── user_routes.py     # Routes utilisateurs
│   └── chat.py            # Routes de chat
├── services/
│   ├── auth.py            # Service d'authentification
│   └── vector_search.py   # Service de recherche vectorielle
├── migrate_movies.py      # Script de migration des données
└── test_setup.py          # Script de test de configuration
```

## 🚀 Installation et Configuration

### 1. Prérequis

- Python 3.11+
- Docker et Docker Compose
- MongoDB (via Docker)

### 2. Installation des dépendances

```bash
pip install -r requirements.txt
```

### 3. Configuration de l'environnement

```bash
# Copier le fichier d'environnement
cp .env.example .env

# Éditer les variables d'environnement
nano .env
```

### 4. Lancement avec Docker

```bash
# Démarrer MongoDB et l'API
docker-compose up -d

# Vérifier les logs
docker-compose logs -f
```

### 5. Test de la configuration

```bash
# Exécuter les tests de configuration
python test_setup.py
```

## 📊 Base de Données

### Structure MongoDB

#### Collection `users`
```javascript
{
  "_id": ObjectId,
  "email": "user@example.com",
  "username": "username",
  "first_name": "John",
  "last_name": "Doe",
  "hashed_password": "...",
  "is_active": true,
  "created_at": ISODate,
  "following": [ObjectId, ...],
  "followers": [ObjectId, ...],
  "liked_movies": [ObjectId, ...],
  "disliked_movies": [ObjectId, ...]
}
```

#### Collection `movies`
```javascript
{
  "_id": ObjectId,
  "tmdb_id": 12345,
  "title": "Film Title",
  "overview": "Description...",
  "release_date": "2023-01-01",
  "genres": ["Action", "Drama"],
  "vote_average": 8.5,
  "popularity": 75.5,
  // Embeddings vectoriels
  "title_embedding": [0.1, 0.2, ...],
  "overview_embedding": [0.3, 0.4, ...],
  "combined_embedding": [0.5, 0.6, ...],
  // Métadonnées
  "poster_path": "/path/to/poster.jpg",
  "backdrop_path": "/path/to/backdrop.jpg"
}
```

### Index de Performance

```javascript
// Index utilisateurs
db.users.createIndex({ "email": 1 }, { unique: true })
db.users.createIndex({ "username": 1 }, { unique: true })

// Index films
db.movies.createIndex({ "tmdb_id": 1 }, { unique: true })
db.movies.createIndex({ "title": "text", "overview": "text", "genres": "text" })
db.movies.createIndex({ "genres": 1 })
db.movies.createIndex({ "vote_average": -1 })
db.movies.createIndex({ "popularity": -1 })
```

## 🔍 Recherche Vectorielle

### Génération d'Embeddings

Le système utilise **Sentence Transformers** (`all-MiniLM-L6-v2`) pour générer des embeddings vectoriels :

```python
# Génération automatique d'embeddings
embeddings = await vector_search_service.generate_movie_embeddings(movie)
```

### Types de Recherche

1. **Recherche textuelle** - Similarité cosinus avec la requête
2. **Films similaires** - Basé sur les embeddings des films
3. **Recommandations personnalisées** - Profil utilisateur basé sur les films aimés

## 🛡 Authentification

### JWT Token

```bash
# Login
POST /auth/login
{
  "email": "user@example.com",
  "password": "password"
}

# Réponse
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": { ... }
}
```

### Utilisation du Token

```bash
# Headers pour les requêtes authentifiées
Authorization: Bearer eyJ...
```

## 📡 API Endpoints

### Authentification
- `POST /auth/register` - Créer un compte
- `POST /auth/login` - Se connecter  
- `GET /auth/me` - Profil utilisateur
- `POST /auth/refresh` - Renouveler le token

### Films
- `GET /movies/search?query=...` - Recherche vectorielle
- `GET /movies/{id}/similar` - Films similaires
- `GET /movies/recommendations` - Recommandations personnalisées
- `POST /movies/swipe` - Liker/Disliker un film
- `GET /movies/popular` - Films populaires
- `POST /movies/bulk-insert` - Import en masse

### Utilisateurs
- `GET /users/profile` - Profil utilisateur
- `PUT /users/profile` - Modifier le profil
- `POST /users/follow` - Suivre un utilisateur

## 🔄 Migration des Données

### Import des Films TMDB

```bash
# Exécuter le script de migration
python migrate_movies.py
```

Le script :
1. Lit les données depuis `db/tmdb_movies_for_embedding3.json`
2. Génère les embeddings vectoriels
3. Insère/met à jour les films dans MongoDB
4. Traite par batches pour optimiser les performances

### Format des Données

```json
{
  "id": 12345,
  "title": "Film Title",
  "overview": "Description...",
  "release_date": "2023-01-01",
  "genres": [{"name": "Action"}, {"name": "Drama"}],
  "vote_average": 8.5,
  "popularity": 75.5,
  "poster_path": "/path.jpg"
}
```

## 🐳 Docker

### Services

```yaml
services:
  mongodb:      # Base de données MongoDB avec replica set
  mongo-init:   # Initialisation du replica set
  api:          # API FastAPI
```

### Commandes Utiles

```bash
# Démarrer les services
docker-compose up -d

# Voir les logs
docker-compose logs -f api

# Redémarrer l'API
docker-compose restart api

# Accéder à MongoDB
docker exec -it mongodb-vector-db mongosh -u root -p rootpassword
```

## 🧪 Tests

### Tests Automatisés

```bash
# Exécuter tous les tests
python test_setup.py
```

Tests inclus :
- ✅ Connexion MongoDB
- ✅ Génération d'embeddings
- ✅ Opérations CRUD films
- ✅ Opérations utilisateurs
- ✅ Recherche vectorielle

### Tests Manuels

```bash
# Test de l'API
curl http://localhost:8000/

# Test de recherche
curl "http://localhost:8000/movies/search?query=action%20hero"
```

## 📈 Performance

### Optimisations

- **Index MongoDB** pour les requêtes fréquentes
- **Traitement par batches** pour l'import des données
- **Cache des embeddings** pour éviter la regeneration
- **Connexions asynchrones** partout

### Métriques

- Temps de génération d'embedding : ~100ms par film
- Recherche vectorielle : ~50ms pour 1000 films
- Import en masse : ~1000 films/minute

## 🔧 Développement

### Structure du Code

```python
# Modèle avec Beanie
class Movie(Document):
    title: str
    combined_embedding: Optional[List[float]] = None
    
    class Settings:
        name = "movies"

# Service vectoriel
await vector_search_service.search_movies_by_text("query")

# Route FastAPI
@router.get("/search")
async def search_movies(query: str):
    return await vector_search_service.search_movies_by_text(query)
```

### Bonnes Pratiques

- Utiliser `async/await` partout
- Valider les données avec Pydantic
- Gérer les erreurs proprement
- Logger les opérations importantes
- Tester les nouvelles fonctionnalités

## 🚨 Dépannage

### Problèmes Courants

1. **Connexion MongoDB échoue**
   ```bash
   # Vérifier que MongoDB est démarré
   docker-compose ps
   ```

2. **Import d'embeddings lent**
   ```bash
   # Réduire la taille des batches
   batch_size = 50  # au lieu de 100
   ```

3. **Erreur de mémoire**
   ```bash
   # Augmenter la mémoire Docker
   # Docker Desktop > Settings > Resources
   ```

### Logs Utiles

```bash
# Logs de l'API
docker-compose logs api

# Logs MongoDB
docker-compose logs mongodb

# Logs en temps réel
docker-compose logs -f
```

## 🌟 Fonctionnalités Avancées

### Recherche Hybride

Combinaison de :
- Recherche vectorielle (similarité sémantique)
- Recherche textuelle (MongoDB text search)
- Filtres (genres, notes, date)

### Recommandations Intelligentes

- Profil utilisateur basé sur les films aimés
- Algorithme de similarité cosinus
- Exclusion des films déjà vus

### Système de Suivi

- Suivre d'autres utilisateurs
- Recommandations basées sur le réseau social
- Partage de listes de films

## 📝 TODO

- [ ] Recherche vectorielle avec MongoDB Atlas Vector Search
- [ ] Cache Redis pour les recommandations
- [ ] API de streaming pour les gros datasets
- [ ] Tests d'intégration complets
- [ ] Déploiement en production
- [ ] Monitoring et métriques

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commiter les changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

MIT License - voir le fichier LICENSE pour plus de détails. API

API backend pour l'application MovieMind, construite avec FastAPI.

## 🚀 Fonctionnalités

### Système d'authentification complet
- ✅ Enregistrement d'utilisateurs
- ✅ Connexion/Déconnexion
- ✅ Tokens JWT sécurisés
- ✅ Routes protégées
- ✅ Renouvellement de tokens
- ✅ Gestion des utilisateurs mockés (en attente de la base de données)

## 📋 Prérequis

- Python 3.8+
- pip

## 🛠️ Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancer le serveur :
```bash
uvicorn main:app --reload
```

L'API sera disponible sur `http://localhost:8000`

## 📚 Documentation

- **Swagger UI** : `http://localhost:8000/docs`
- **ReDoc** : `http://localhost:8000/redoc`

## 🔐 Authentification

### Utilisateurs de test (mockés)

Le système utilise actuellement des utilisateurs mockés. Voici les comptes disponibles :

| Email | Mot de passe | Rôle |
|-------|-------------|------|
| `john.doe@example.com` | `secret` | Utilisateur |
| `jane.smith@example.com` | `secret` | Utilisateur |
| `admin@moviemind.com` | `secret` | Admin |

### Endpoints d'authentification

- `POST /auth/register` - Créer un nouveau compte
- `POST /auth/login` - Se connecter
- `GET /auth/me` - Informations de l'utilisateur connecté
- `POST /auth/refresh` - Renouveler le token
- `POST /auth/logout` - Se déconnecter

### Utilisation des tokens

1. **Connexion** : Envoyez vos identifiants à `/auth/login`
2. **Récupération du token** : L'API retourne un token JWT
3. **Utilisation** : Ajoutez le header `Authorization: Bearer YOUR_TOKEN` à vos requêtes
4. **Expiration** : Les tokens expirent après 30 minutes

### Exemple d'utilisation

```bash
# 1. Se connecter
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "john.doe@example.com", "password": "secret"}'

# 2. Utiliser le token reçu
curl -X GET "http://localhost:8000/protected/profile" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🛡️ Routes protégées (exemples)

- `GET /protected/profile` - Profil de l'utilisateur connecté
- `GET /protected/movies/recommendations` - Recommandations personnalisées

## 🧪 Tests

Utilisez le fichier `test_main.http` avec l'extension REST Client de VS Code pour tester facilement tous les endpoints.

## 🔧 Configuration

### Variables d'environnement importantes

- `SECRET_KEY` : Clé secrète pour signer les tokens JWT (changez-la en production !)
- `ACCESS_TOKEN_EXPIRE_MINUTES` : Durée de vie des tokens (défaut : 30 min)

### Sécurité

⚠️ **Important pour la production** :
- Changez la `SECRET_KEY` dans `services/auth.py`
- Utilisez des variables d'environnement pour les secrets
- Configurez HTTPS
- Limitez les origines CORS

## 📁 Structure du projet

```
MovieMindBack/
├── main.py                 # Point d'entrée de l'application
├── requirements.txt        # Dépendances Python
├── test_main.http         # Tests des endpoints
├── routes/
│   ├── auth.py            # Routes d'authentification
│   └── example.py         # Exemples de routes protégées
└── services/
    └── auth.py            # Service d'authentification et gestion des tokens
```

## 🚧 À venir

- [ ] Intégration avec une vraie base de données
- [ ] Gestion des rôles et permissions
- [ ] Réinitialisation de mot de passe
- [ ] Validation email
- [ ] Rate limiting
- [ ] Logging avancé