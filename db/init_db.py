from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.types import JSON
import os

# Create db directory if it doesn't exist
os.makedirs("db", exist_ok=True)

DATABASE_URL = "sqlite+aiosqlite:///db/test.db"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


follow_table = Table(
    "follows",
    Base.metadata,
    Column("follower_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("followed_id", Integer, ForeignKey("users.id"), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String, nullable=False)
    prenom = Column(String, nullable=False)
    embedding = Column(JSON)

    followers = relationship(
        "User",
        secondary=follow_table,
        primaryjoin=id == follow_table.c.followed_id,
        secondaryjoin=id == follow_table.c.follower_id,
        backref="following"
    )

# Dependency
async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

# Initialize database tables
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)