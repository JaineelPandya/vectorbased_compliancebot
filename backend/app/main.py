import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

from backend.app.config import settings
from backend.app.database import async_engine, Base
from backend.app.api.endpoints import router as api_router

# Setup application-wide logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    logger.info("Initializing database tables...")
    try:
        from sqlalchemy import text
        async with async_engine.begin() as conn:
            # Create PostgreSQL database tables if they do not exist
            await conn.run_sync(Base.metadata.create_all)
            # Run schema self-healing migrations to add new columns if they don't exist
            await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS keywords JSONB"))
            await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS topics JSONB"))
            await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS entities JSONB"))
            await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS financial_terms JSONB"))
            await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS circular_type VARCHAR"))
            # P0: subject field for all documents
            await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS subject TEXT"))
        logger.info("Database tables initialized and migrated successfully.")
    except Exception as e:
        logger.error(f"Database table initialization failed: {e}. Ensure PostgreSQL is running.")

    # P1: Warm up reranker at startup so the model is ready on first query
    try:
        from backend.app.services.reranker import reranker
        _ = reranker.model   # triggers model load now, not on first request
        logger.info("Reranker model pre-loaded at startup.")
    except Exception as e:
        logger.warning(f"Reranker warm-up failed (non-fatal): {e}")
        
    yield
    logger.info("Shutting down API service...")

app = FastAPI(
    title=settings.app.name,
    description="Production-Ready Agentic PDF Ingestion and Hybrid Semantic Search Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integrations (React + Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Endpoints Router
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug
    )
