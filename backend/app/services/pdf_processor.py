import os
import hashlib
import logging
import uuid
import fitz  # PyMuPDF
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from backend.app.config import settings
from backend.app.models import Document, DocumentPage, ProcessingLog
from backend.app.services.embeddings import embedding_service
from backend.app.services.vector_store import qdrant_store
from backend.app.services.search_store import search_store
from backend.app.services.vision_pipeline import vision_pipeline

logger = logging.getLogger("app.pdf_processor")

class PDFProcessor:
    def __init__(self):
        # Docling fallback support
        self.docling_available = False
        try:
            from docling.document_converter import DocumentConverter
            self.docling_converter = DocumentConverter()
            self.docling_available = True
            logger.info("Docling parser loaded successfully.")
        except Exception as e:
            logger.warning(f"Docling could not be loaded: {e}. PyMuPDF will act as the primary parser.")

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculates the SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def log_step(
        self, 
        db: AsyncSession, 
        doc_id: uuid.UUID, 
        step: str, 
        status: str, 
        message: Optional[str] = None
    ):
        """Logs ingestion step status to the database."""
        log_entry = ProcessingLog(
            document_id=doc_id,
            step=step,
            status=status,
            message=message
        )
        db.add(log_entry)
        await db.commit()
        logger.info(f"[{doc_id}] Step {step} - {status}: {message}")

    def classify_page(self, page: fitz.Page) -> Tuple[str, float]:
        """
        Classifies a page into text, scanned, graph, table, or mixed.
        Returns the type string and the classifier's confidence score.
        """
        text = page.get_text().strip()
        images = page.get_images()
        drawings = page.get_drawings()
        
        char_count = len(text)
        image_count = len(images)
        drawing_count = len(drawings)

        # 1. Scanned Page Heuristic
        if char_count < 100 and image_count >= 1:
            return "scanned", 0.95
        
        # 2. Table Heuristic (Inspecting page text for grid/column/tabular patterns)
        table_keywords = ["table", "annexure", "schedule", "statement of", "balance sheet", "particulars", "sr. no"]
        has_table_keywords = any(kw in text.lower() for kw in table_keywords)
        
        # Checking for tabular column spaces
        lines = text.split("\n")
        aligned_columns = 0
        for line in lines:
            if len(line.split("   ")) > 2 or "\t" in line:
                aligned_columns += 1
        
        if (aligned_columns > 3 or has_table_keywords) and drawing_count > 5:
            return "table", 0.85

        # 3. Graph Heuristic (Inspecting drawings/paths commonly present in charts)
        # Bar charts/plots use vector paths
        if drawing_count > 30 and not has_table_keywords:
            return "graph", 0.80

        # 4. Mixed Page
        if char_count >= 100 and (image_count >= 1 or drawing_count > 10):
            return "mixed", 0.75

        # 5. Text Page (Default fallback)
        return "text", 0.90

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Chunks page text into overlapping windows."""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        # Re-assemble using a word-based slider
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + chunk_size >= len(words):
                break
            i += (chunk_size - overlap)
        
        return chunks

    async def ingest_pdf(
        self,
        file_path: str,
        name: str,
        circular_number: Optional[str],
        issue_date: Optional[date],
        department: Optional[str],
        tags: Optional[List[str]],
        db: AsyncSession
    ) -> Optional[Document]:
        """
        Ingests document through the parsing, classification, embedding, and indexing pipeline.
        Handles circular versioning and indexing in both vector stores (Qdrant) and search indexes (Elasticsearch).
        """
        # 1. Calculate File Hash
        file_hash = self.calculate_file_hash(file_path)
        
        # 2. Check for exact duplicate hash
        stmt = select(Document).where(Document.hash == file_hash)
        res = await db.execute(stmt)
        existing_doc = res.scalar_one_or_none()
        
        if existing_doc:
            logger.info(f"Duplicate file uploaded: {name} (ID: {existing_doc.id}). Skipping.")
            return existing_doc

        # 3. Resolve versioning
        version = 1
        if circular_number:
            # Query active documents with same circular number to archive them
            stmt_circ = select(Document).where(Document.circular_number == circular_number)
            res_circ = await db.execute(stmt_circ)
            matched_docs = res_circ.scalars().all()
            
            if matched_docs:
                # Get max version
                max_version = max(d.version for d in matched_docs)
                version = max_version + 1
                
                # Archive older versions
                await db.execute(
                    update(Document)
                    .where(Document.circular_number == circular_number)
                    .values(status="archived")
                )
                await db.commit()

        # 4. Create Document Entry
        doc = Document(
            id=uuid.uuid4(),
            name=name,
            circular_number=circular_number,
            hash=file_hash,
            version=version,
            status="processing",
            issue_date=issue_date or date.today(),
            department=department or "Unknown",
            tags=tags or []
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        doc_id_str = str(doc.id)
        
        await self.log_step(db, doc.id, "UPLOAD", "SUCCESS", f"Created document metadata. Version: {version}")

        # Ensure index setup in Elasticsearch
        await search_store.init_indices()

        try:
            # 5. Open PDF and extract pages
            doc_fitz = fitz.open(file_path)
            num_pages = len(doc_fitz)
            await self.log_step(db, doc.id, "PARSING", "SUCCESS", f"Opened PDF with {num_pages} pages.")

            all_es_chunks = []
            
            for page_idx in range(num_pages):
                page = doc_fitz[page_idx]
                page_num = page_idx + 1
                
                # 5a. Classify Page
                classification, confidence = self.classify_page(page)
                
                vision_summary = None
                vision_extracted_values = None
                
                # 5b. Check if Vision is needed (scanned, table, graph, mixed)
                if classification in ["scanned", "table", "graph"]:
                    await self.log_step(db, doc.id, "VISION", "RUNNING", f"Triggering Qwen3-VL for page {page_num} ({classification})")
                    # Render page as image bytes for vision pipeline
                    pix = page.get_pixmap(dpi=150)
                    image_bytes = pix.tobytes("png")
                    
                    vision_result = await vision_pipeline.analyze_page_image(
                        image_bytes=image_bytes,
                        page_type=classification,
                        page_num=page_num
                    )
                    
                    vision_summary = vision_result.get("summary")
                    vision_extracted_values = vision_result.get("extracted_values")
                    confidence = vision_result.get("confidence", confidence)

                # Store Page structure in Database
                db_page = DocumentPage(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    page_number=page_num,
                    classification=classification,
                    confidence=confidence,
                    vision_summary=vision_summary,
                    vision_extracted_values=vision_extracted_values
                )
                db.add(db_page)
                await db.commit()

                # 5c. Chunk page text + vision summaries
                raw_text = page.get_text()
                combined_text_for_embedding = raw_text
                
                if vision_summary:
                    combined_text_for_embedding += f"\n[Vision Analysis Summary]\n{vision_summary}"
                
                chunks = self.chunk_text(combined_text_for_embedding)
                if not chunks and vision_summary:
                    chunks = [vision_summary]

                # 5d. Map collection target
                collection_map = {
                    "text": "text_chunks",
                    "table": "table_chunks",
                    "graph": "graph_chunks",
                    "scanned": "scan_chunks",
                    "mixed": "text_chunks"
                }
                collection_name = collection_map.get(classification, "text_chunks")

                # Generate Embeddings & Upsert to Qdrant
                if chunks:
                    embeddings = await embedding_service.get_embeddings(chunks)
                    qdrant_points = []
                    
                    for chunk_idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
                        chunk_id = str(uuid.uuid5(doc.id, f"page_{page_num}_chunk_{chunk_idx}"))
                        
                        payload = {
                            "doc_id": doc_id_str,
                            "page": page_num,
                            "section": f"Page {page_num} Section {chunk_idx + 1}",
                            "document_version": version,
                            "upload_date": str(doc.issue_date),
                            "text": chunk_text
                        }
                        
                        qdrant_points.append({
                            "id": chunk_id,
                            "vector": vector,
                            "payload": payload
                        })
                        
                        # Accumulate chunk dictionary for Elasticsearch bulk/indexing
                        all_es_chunks.append({
                            "chunk_id": chunk_id,
                            "doc_id": doc_id_str,
                            "page": page_num,
                            "section": f"Page {page_num} Section {chunk_idx + 1}",
                            "text": chunk_text,
                            "circular_number": circular_number,
                            "issue_date": str(doc.issue_date),
                            "version": version,
                            "department": doc.department,
                            "tags": doc.tags,
                            "status": doc.status
                        })
                    
                    # Upsert points into Qdrant
                    await qdrant_store.upsert_chunks(collection_name, qdrant_points)

            # 6. Index Document Metadata in Elasticsearch
            es_metadata = {
                "name": doc.name,
                "circular_number": doc.circular_number,
                "issue_date": str(doc.issue_date),
                "version": doc.version,
                "department": doc.department,
                "tags": doc.tags,
                "status": "active",
                "uploaded_at": doc.uploaded_at.isoformat()
            }
            await search_store.index_document_metadata(doc_id_str, es_metadata)

            # Index individual chunks in Elasticsearch to support keyword BM25 search
            if all_es_chunks:
                await search_store.index_chunks(all_es_chunks)

            # 7. Update Document Status to Active
            doc.status = "active"
            await db.commit()
            await self.log_step(db, doc.id, "COMPLETE", "SUCCESS", "Document fully processed and indexed.")
            return doc

        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            doc.status = "failed"
            await db.commit()
            await self.log_step(db, doc.id, "PROCESS", "FAILED", f"Critical processing failure: {str(e)}")
            return doc

pdf_processor = PDFProcessor()
