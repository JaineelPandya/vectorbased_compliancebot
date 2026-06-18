import logging
import uuid
from typing import List, Optional
from datetime import date
from backend.app.services.redis_cache import redis_cache
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from backend.app.database import get_db
from backend.app.models import Document, DocumentPage, AgentTrace, ProcessingLog
from backend.app.schemas import (
    DocumentResponse, 
    QueryRequest, 
    QueryResponse, 
    AgentTraceResponse,
    ProcessingLogResponse
)
from backend.app.services.pdf_processor import pdf_processor
from backend.app.services.agent_workflow import run_agent_search
from backend.app.services.vector_store import qdrant_store
from backend.app.services.search_store import search_store
from backend.app.config import settings

logger = logging.getLogger("app.endpoints")
router = APIRouter()

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    circular_number: Optional[str] = Form(None),
    issue_date: Optional[date] = Form(None),
    department: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # Comma-separated list of strings
    db: AsyncSession = Depends(get_db)
):
    """
    API endpoint to upload and process a digital/scanned PDF.
    Generates hashes, version increments, page classifications, visual summaries, embeddings, and indices.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF file uploads are supported.")

    # Save file to upload directory
    file_name = name or file.filename
    temp_path = os.path.join(settings.storage.temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # Write contents to temporary storage path
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        
        # Run Ingestion
        document = await pdf_processor.ingest_pdf(
            file_path=temp_path,
            name=file_name,
            circular_number=circular_number,
            issue_date=issue_date,
            department=department,
            tags=parsed_tags,
            db=db
        )
        
        if not document:
            raise HTTPException(status_code=500, detail="Failed to process document.")
            
        return document
        
    except Exception as e:
        logger.error(f"Upload API execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")
        
    finally:
        # Cleanup temp upload file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/query", response_model=QueryResponse)
async def query_system(payload: QueryRequest):
    """
    Triggers the LangGraph-based multi-agent search agent pipeline.
    Resolves exact constraints, plans, retrieves hybrid vectors, audits facts, and synthesizes answers with citations.
    """
    try:
        result = await run_agent_search(
            query=payload.query,
            session_id=payload.session_id,
            filters=payload.filters
        )
        return result
    except Exception as e:
        logger.error(f"Query API failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search pipeline error: {str(e)}")

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """Lists all uploaded documents and circulars."""
    try:
        stmt = select(Document).order_by(Document.uploaded_at.desc())
        res = await db.execute(stmt)
        return res.scalars().all()
    except Exception as e:
        logger.error(f"List documents API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/document/{id}", response_model=DocumentResponse)
async def get_document(id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Retrieves metadata of a single document."""
    stmt = select(Document).where(Document.id == id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@router.delete("/document/{id}", status_code=status.HTTP_200_OK)
async def delete_document(id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Deletes document metadata and chunks from PostgreSQL, Qdrant vector spaces, and Elasticsearch indexes.
    """
    stmt = select(Document).where(Document.id == id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc_id_str = str(id)
    try:
        # 1. Delete from Qdrant vector db
        await qdrant_store.delete_document_vectors(doc_id_str)
        # 2. Delete from Elasticsearch index
        await search_store.delete_document(doc_id_str)
        # 3. Delete from PostgreSQL database (cascades pages and logs)
        await db.execute(delete(Document).where(Document.id == id))
        await db.commit()
        return {"status": "success", "message": f"Document {id} and all related vectors/indexes deleted successfully."}
    except Exception as e:
        logger.error(f"Deletion failed for document {id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/agent-trace", response_model=List[AgentTraceResponse])
async def get_agent_trace(session_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieves execution steps and input/output payloads of agents for a given query session."""
    try:
        stmt = select(AgentTrace).where(AgentTrace.session_id == session_id).order_by(AgentTrace.timestamp.asc())
        res = await db.execute(stmt)
        return res.scalars().all()
    except Exception as e:
        logger.error(f"Agent trace retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/citations")
async def get_citations(session_id: str, db: AsyncSession = Depends(get_db)):
    """Endpoint displaying citation lists mapped from traces or current session."""
    try:
        stmt = select(AgentTrace).where(AgentTrace.session_id == session_id, AgentTrace.step_name == "Citation")
        res = await db.execute(stmt)
        trace_record = res.scalars().first()
        if not trace_record:
            return []
        return trace_record.output_state.get("citation_count", 0)
    except Exception as e:
        logger.error(f"Citations retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Validates connectivity across database integrations and system services."""
    health_status = {
        "status": "healthy",
        "database": "connected",
        "redis": "connected" if redis_cache.redis_client else "in_memory_fallback",
        "qdrant": "connected" if qdrant_store.client else "disconnected",
        "elasticsearch": "connected" if search_store.client else "disconnected"
    }
    
    if qdrant_store.client is None or search_store.client is None:
        health_status["status"] = "degraded"
        
    return health_status

import os
