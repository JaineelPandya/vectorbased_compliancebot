import logging
import uuid
import os
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
    upload_name = name.strip() if name and name.strip() else None
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
            name=upload_name,
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

from backend.app.services.reranker import reranker

@router.get("/debug/retrieval")
async def debug_retrieval(query: str):
    """Debug endpoint to see raw vs reranked chunks directly."""
    try:
        from backend.app.services.embeddings import embedding_service
        vector = await embedding_service.get_embedding(query)
        
        qdrant_results = await qdrant_store.search_collection(vector, limit=50)
        
        es_results = await search_store.search_chunks(query, limit=20)
        
        merged = []
        seen = set()
        for r in qdrant_results + es_results:
            text = r.get("payload", {}).get("text", "")
            import hashlib
            h = hashlib.sha256(text.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                merged.append(r.get("payload", {}))
                
        reranked = reranker.rerank(query, merged, top_k=8)
        
        return {
            "query": query,
            "qdrant_results": len(qdrant_results),
            "elastic_results": len(es_results),
            "reranked_results": reranked
        }
    except Exception as e:
        logger.error(f"Debug retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/chunk/{doc_id}")
async def debug_chunk(doc_id: str):
    """Debug endpoint to verify chunking accuracy and metadata payload in ES."""
    try:
        # Search Elasticsearch for all chunks belonging to this doc_id
        results = await search_store.search_chunks("", limit=100, filters={"doc_id": doc_id})
        return results
    except Exception as e:
        logger.error(f"Debug chunk failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/vague-documents")
async def debug_vague_documents():
    """Debug endpoint to identify documents with vague or missing names."""
    try:
        results = await search_store.search_chunks("", limit=1000)
        vague_docs = {}
        
        for result in results:
            payload = result.get("payload", {})
            title = payload.get("title")
            doc_id = payload.get("doc_id")
            
            # Identify vague document titles
            if not title or title == "Unknown Title" or len(title.strip()) == 0:
                if doc_id not in vague_docs:
                    vague_docs[doc_id] = {
                        "title": title or "NOT PROVIDED",
                        "chunk_count": 0,
                        "pages": set()
                    }
                vague_docs[doc_id]["chunk_count"] += 1
                vague_docs[doc_id]["pages"].add(payload.get("page_number"))
        
        # Convert sets to lists for JSON serialization
        for doc_id in vague_docs:
            vague_docs[doc_id]["pages"] = sorted(list(vague_docs[doc_id]["pages"]))
        
        return {
            "vague_documents_found": len(vague_docs),
            "details": vague_docs,
            "recommendation": "Please review documents with missing titles. Titles are generated from first-page content and should be descriptive of the document content."
        }
    except Exception as e:
        logger.error(f"Debug vague documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@router.post("/document/{id}/reindex-metadata")
async def reindex_document_metadata(id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Re-runs LLM metadata extraction (keywords, topics, entities, financial_terms) 
    for an existing document and updates PostgreSQL, Qdrant, and Elasticsearch.
    Use this when a document's keywords are missing or incorrect.
    """
    stmt = select(Document).where(Document.id == id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    try:
        # Get representative text chunks from Elasticsearch
        es_res = await search_store.client.search(
            index=search_store.chunk_index,
            body={
                'query': {
                    'bool': {
                        'filter': [{'term': {'doc_id': str(id)}}],
                        'must_not': [
                            {'match_phrase': {'text': 'Fallback extraction'}},
                            {'match_phrase': {'text': 'Go to Index'}},
                            {'match_phrase': {'text': 'Continuation Sheet'}},
                        ]
                    }
                },
                'sort': [{'page_number': {'order': 'asc'}}],
                '_source': ['text'],
                'size': 20
            }
        )
        
        hits = es_res['hits']['hits']
        chunks = [hit['_source'].get('text', '') for hit in hits
                 if len(hit['_source'].get('text', '').split()) > 20]
        
        if not chunks:
            raise HTTPException(status_code=422, detail="No meaningful text chunks found for this document. Cannot extract metadata.")
        
        sample_text = '\n\n'.join(chunks[:10])
        
        # Run metadata extraction with LLM
        meta = await pdf_processor.extract_document_metadata(sample_text)
        
        # Update PostgreSQL
        doc.keywords = meta.get('keywords') or []
        doc.topics = meta.get('topics') or []
        doc.entities = meta.get('entities') or []
        doc.financial_terms = meta.get('financial_terms') or []
        doc.circular_type = meta.get('circular_type') or 'Circular'
        await db.commit()
        await db.refresh(doc)
        
        # Update Qdrant payloads
        qdrant_updated = 0
        from qdrant_client.http import models as qmodels
        offset = None
        while True:
            scroll_res = qdrant_store.client.scroll(
                collection_name='document_chunks',
                scroll_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=str(id)))]
                ),
                limit=200,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            points, next_offset = scroll_res
            if not points:
                break
            qdrant_store.client.set_payload(
                collection_name='document_chunks',
                payload={
                    'keywords': doc.keywords,
                    'topics': doc.topics,
                    'entities': doc.entities,
                    'financial_terms': doc.financial_terms,
                    'circular_type': doc.circular_type,
                },
                points=[p.id for p in points]
            )
            qdrant_updated += len(points)
            if next_offset is None:
                break
            offset = next_offset
        
        # Update Elasticsearch chunks
        es_update = await search_store.client.update_by_query(
            index=search_store.chunk_index,
            body={
                'query': {'term': {'doc_id': str(id)}},
                'script': {
                    'source': 'ctx._source.keywords = params.keywords; ctx._source.topics = params.topics; ctx._source.entities = params.entities; ctx._source.financial_terms = params.financial_terms; ctx._source.circular_type = params.circular_type;',
                    'params': {
                        'keywords': doc.keywords,
                        'topics': doc.topics,
                        'entities': doc.entities,
                        'financial_terms': doc.financial_terms,
                        'circular_type': doc.circular_type,
                    }
                }
            }
        )
        
        return {
            "status": "success",
            "doc_id": str(id),
            "keywords": doc.keywords,
            "topics": doc.topics,
            "entities": doc.entities,
            "financial_terms": doc.financial_terms,
            "circular_type": doc.circular_type,
            "qdrant_points_updated": qdrant_updated,
            "es_chunks_updated": es_update.get('updated', 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex metadata failed for doc {id}: {e}")
        raise HTTPException(status_code=500, detail=f"Metadata reindex failed: {str(e)}")


@router.post("/database/repair")
async def repair_database():
    """
    Cleans up boilerplate chunks (Go to Index, Continuation Sheet) from Elasticsearch
    and refreshes the index. Run this if document retrieval returns garbage content.
    """
    try:
        BOILERPLATE_PHRASES = ["Go to Index", "Continuation Sheet", "P a g e", "Part – A"]
        total_deleted = 0
        for phrase in BOILERPLATE_PHRASES:
            del_res = await search_store.client.delete_by_query(
                index=search_store.chunk_index,
                body={'query': {'match_phrase': {'text': phrase}}}
            )
            deleted = del_res.get('deleted', 0)
            total_deleted += deleted
            if deleted > 0:
                logger.info(f"Repair: Deleted {deleted} ES chunks containing '{phrase}'")
        
        await search_store.client.indices.refresh(index=search_store.chunk_index)
        
        es_count = await search_store.client.count(index=search_store.chunk_index)
        
        return {
            "status": "success",
            "boilerplate_chunks_deleted": total_deleted,
            "remaining_chunks": es_count['count'],
            "message": f"Removed {total_deleted} boilerplate chunks. {es_count['count']} valid chunks remain."
        }
    except Exception as e:
        logger.error(f"Database repair failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database repair failed: {str(e)}")


# ---------------------------------------------------------------------------
# Reindex endpoints (full pipeline re-ingest with new chunking schema)
# ---------------------------------------------------------------------------

@router.post("/documents/{id}/reindex")
async def reindex_document(id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Full re-ingest of an existing document:
      1. Delete stale Qdrant vectors
      2. Delete stale Elasticsearch chunks
      3. Re-parse PDF from stored file path (matched by SHA-256 hash)
      4. Generate new hierarchical chunks with subject-boosted embeddings
      5. Insert into Qdrant + Elasticsearch

    Response includes per-type chunk counts:
      { doc_id, status, chunks_created, table_chunks, graph_chunks }
    """
    stmt = select(Document).where(Document.id == id)
    res = await db.execute(stmt)
    doc = res.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc_id_str = str(id)
    upload_dir = settings.storage.upload_dir

    # Locate stored PDF by SHA-256 hash match
    pdf_path: Optional[str] = None
    if os.path.isdir(upload_dir):
        for fname in os.listdir(upload_dir):
            if fname.endswith(".pdf"):
                full = os.path.join(upload_dir, fname)
                try:
                    if pdf_processor.calculate_file_hash(full) == doc.hash:
                        pdf_path = full
                        break
                except Exception:
                    continue

    if not pdf_path:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Original PDF file not found in upload directory for document {id}. "
                "The file may have been cleaned up after initial ingest. "
                "Please re-upload the document instead."
            )
        )

    try:
        doc.status = "processing"
        await db.commit()

        # 1. Delete stale vectors from Qdrant
        await qdrant_store.delete_document_vectors(doc_id_str)
        logger.info(f"[Reindex] Deleted Qdrant vectors for doc_id={doc_id_str}")

        # 2. Delete stale chunks + metadata from Elasticsearch
        await search_store.delete_document(doc_id_str)
        logger.info(f"[Reindex] Deleted ES records for doc_id={doc_id_str}")

        # 3. Delete existing DocumentPage records so they are recreated cleanly
        from sqlalchemy import delete as sa_delete
        await db.execute(sa_delete(DocumentPage).where(DocumentPage.document_id == id))
        await db.commit()

        # 4. Re-run full ingest pipeline
        processed_doc = await pdf_processor.ingest_pdf(
            file_path=pdf_path,
            name=doc.name,
            circular_number=doc.circular_number,
            issue_date=doc.issue_date,
            department=doc.department,
            tags=doc.tags or [],
            db=db
        )

        if not processed_doc:
            raise HTTPException(status_code=500, detail="Reindex pipeline failed.")

        stats = getattr(processed_doc, "_ingest_stats", {})
        return {
            "doc_id":         doc_id_str,
            "status":         "success",
            "chunks_created": stats.get("chunks_created", 0),
            "table_chunks":   stats.get("table_chunks", 0),
            "graph_chunks":   stats.get("graph_chunks", 0),
            "message":        f"Document {id} successfully reindexed with hierarchical chunking."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex failed for document {id}: {e}")
        doc.status = "failed"
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


@router.post("/documents/reindex_all")
async def reindex_all_documents(db: AsyncSession = Depends(get_db)):
    """
    Bulk reindex all active documents.
    Useful after schema migrations (e.g. adding content_type / subject / vision_confidence fields).

    Returns per-document status and aggregate stats.
    Processes documents sequentially to avoid overwhelming LLM and embedding services.
    """
    stmt = select(Document).where(Document.status == "active")
    res = await db.execute(stmt)
    active_docs = res.scalars().all()

    if not active_docs:
        return {
            "status":         "success",
            "docs_found":     0,
            "docs_reindexed": 0,
            "docs_failed":    0,
            "results":        []
        }

    upload_dir = settings.storage.upload_dir
    results:   List[dict] = []
    reindexed  = 0
    failed     = 0

    for doc in active_docs:
        doc_id_str = str(doc.id)

        # Locate PDF by hash
        pdf_path: Optional[str] = None
        if os.path.isdir(upload_dir):
            for fname in os.listdir(upload_dir):
                if fname.endswith(".pdf"):
                    full = os.path.join(upload_dir, fname)
                    try:
                        if pdf_processor.calculate_file_hash(full) == doc.hash:
                            pdf_path = full
                            break
                    except Exception:
                        continue

        if not pdf_path:
            logger.warning(f"[Reindex All] PDF not found for doc_id={doc_id_str} — skipping.")
            results.append({
                "doc_id": doc_id_str,
                "name":   doc.name,
                "status": "skipped",
                "reason": "PDF file not found in upload directory"
            })
            failed += 1
            continue

        try:
            # Delete stale data
            await qdrant_store.delete_document_vectors(doc_id_str)
            await search_store.delete_document(doc_id_str)
            from sqlalchemy import delete as sa_delete
            await db.execute(sa_delete(DocumentPage).where(DocumentPage.document_id == doc.id))
            await db.commit()

            # Re-ingest
            processed = await pdf_processor.ingest_pdf(
                file_path=pdf_path,
                name=doc.name,
                circular_number=doc.circular_number,
                issue_date=doc.issue_date,
                department=doc.department,
                tags=doc.tags or [],
                db=db
            )
            stats = getattr(processed, "_ingest_stats", {}) if processed else {}
            results.append({
                "doc_id":         doc_id_str,
                "name":           doc.name,
                "status":         "success",
                "chunks_created": stats.get("chunks_created", 0),
                "table_chunks":   stats.get("table_chunks", 0),
                "graph_chunks":   stats.get("graph_chunks", 0)
            })
            reindexed += 1
            logger.info(f"[Reindex All] Successfully reindexed doc_id={doc_id_str}")

        except Exception as e:
            logger.error(f"[Reindex All] Failed for doc_id={doc_id_str}: {e}")
            results.append({
                "doc_id": doc_id_str,
                "name":   doc.name,
                "status": "failed",
                "reason": str(e)
            })
            failed += 1

    return {
        "status":         "complete",
        "docs_found":     len(active_docs),
        "docs_reindexed": reindexed,
        "docs_failed":    failed,
        "results":        results
    }
