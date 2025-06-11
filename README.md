# MovieMind Backend API

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