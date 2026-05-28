import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from database import setup_database, SessionLocal, User
from routes.query_route import router as query_router
from routes.pdf_route   import router as pdf_router
from config import PDF_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(PDF_DIR, exist_ok=True)
    setup_database()
    print("Server ready!")
    yield


app = FastAPI(
    title       = "AI Education Assistant",
    version     = "2.0.0",
    lifespan    = lifespan
)

app.include_router(query_router)
app.include_router(pdf_router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/users", tags=["Health"])
def get_users():
    db    = SessionLocal()
    users = db.query(User).all()
    db.close()
    return [{"id": u.id, "name": u.name, "role": u.role} for u in users]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)