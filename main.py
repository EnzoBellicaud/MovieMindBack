from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.auth import router as auth_router
from routes.example import router as protected_router

app = FastAPI(
    title="MovieMind API",
    description="API pour l'application MovieMind",
    version="1.0.0"
)

# Configuration CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # URLs du frontend Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(auth_router)
app.include_router(protected_router)

@app.get("/")
async def root():
    return {
        "message": "MovieMind API", 
        "version": "1.0.0",
        "status": "Running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
