from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Table
from sqlalchemy.types import JSON
from datetime import datetime
import os

# Create db directory if it doesn't exist
os.makedirs("db", exist_ok=True)

DATABASE_URL = "sqlite+aiosqlite:///db/moviemind.db"
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Table d'association pour les followers
follow_table = Table(
    "follows",
    Base.metadata,
    Column("follower_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("followed_id", Integer, ForeignKey("users.id"), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Preferences de films (optionnel pour l'extension future)
    movie_preferences = Column(JSON, nullable=True)

    # Relations pour le système de suivi
    followers = relationship(
        "User",
        secondary=follow_table,
        primaryjoin=id == follow_table.c.followed_id,
        secondaryjoin=id == follow_table.c.follower_id,
        backref="following"
    )

# Dependency pour obtenir la session de base de données
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Initialize database tables
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)