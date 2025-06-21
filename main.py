from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.auth import router as auth_router
from routes.user_routes import router as user_router
from contextlib import asynccontextmanager
from db.init_db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    await init_db()
    yield
    
app = FastAPI(
    title="MovieMind API",
    description="API pour l'application MovieMind - Découverte et recommandation de films",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000","*"],  # URLs du frontend Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(auth_router)
app.include_router(user_router)

@app.get("/")
async def root():
    return {
        "message": "MovieMind API", 
        "version": "1.0.0",
        "status": "Running",
        "description": "API pour la découverte et recommandation de films"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "MovieMind API"}
