# MovieMind Backend 

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
cp .env_example .env

# Éditer les variables d'environnement avec votre clé api Mistral
nano .env
```

### 4. Lancement avec Docker

```bash
# Démarrer MongoDB et l'API
docker-compose up -d

# Vérifier les logs
docker-compose logs -f
```

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

## Lancer le programme
```bash
uvicorn app:main --reload
```