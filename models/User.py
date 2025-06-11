from typing import List

from pydantic import BaseModel


class UserCreate(BaseModel):
    nom: str
    prenom: str
    embedding: List[float]

class UserRead(BaseModel):
    id: int
    nom: str
    prenom: str
    embedding: List[float]

    class Config:
        orm_mode = True