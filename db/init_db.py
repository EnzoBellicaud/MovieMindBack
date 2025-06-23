from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import os
from typing import Optional

# Models imports
from models.User import User
from models.Movie import Movie
from models.Follow import Follow

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://root:rootpassword@localhost:27017")
DATABASE_NAME = "moviemind"

# Global database client
mongodb_client: Optional[AsyncIOMotorClient] = None

async def get_mongodb_client() -> AsyncIOMotorClient:
    """Get MongoDB client instance"""
    global mongodb_client
    if mongodb_client is None:
        mongodb_client = AsyncIOMotorClient(MONGODB_URL)
    return mongodb_client

async def close_mongodb_connection():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client is not None:
        mongodb_client.close()
        mongodb_client = None

async def init_db():
    """Initialize MongoDB database and collections"""
    client = await get_mongodb_client()
      # Initialize beanie with document models
    await init_beanie(
        database=client[DATABASE_NAME],
        document_models=[User, Movie, Follow]
    )
    
    # Create indexes for better performance
    await create_indexes()
    
    print("MongoDB database initialized successfully!")

async def create_indexes():
    """Create database indexes for optimal performance"""
    try:
        # User indexes
        await User.create_index("email", unique=True)
        await User.create_index("username", unique=True)
        
        # Movie indexes
        await Movie.create_index("tmdb_id", unique=True)
        await Movie.create_index("title")
        await Movie.create_index("genres")
        await Movie.create_index("release_date")
          # Vector search index (will be created manually in MongoDB Atlas or with MongoDB 6.0+)
        # For local development, we'll use text search as fallback
        await Movie.create_index([("title", "text"), ("overview", "text"), ("genres", "text")])
        
        # Follow indexes
        await Follow.create_index("follower_id")
        await Follow.create_index("followed_id")
        await Follow.create_index([("follower_id", 1), ("followed_id", 1)], unique=True)  # Éviter les doublons
        
        print("Database indexes created successfully!")
    except Exception as e:
        print(f"Error creating indexes: {e}")

# Dependency to get database instance
async def get_database():
    """Dependency to get database instance"""
    client = await get_mongodb_client()
    return client[DATABASE_NAME]