from fastapi import APIRouter

router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={404: {"description": "Not found"}},
)
@router.get("/", tags=["users"])
async def read_users():
    return [{"username": "John Doe"}]

@router.get("/{username}", tags=["users"])
async def read_user(username: str):
    return {"username": username}