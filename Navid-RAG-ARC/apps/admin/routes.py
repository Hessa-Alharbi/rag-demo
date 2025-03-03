from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def hello_admin():
    return {"message": "Hello Admin!"}