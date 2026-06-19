import os
import re
import math
import hashlib
import logging
import uuid
import fitz  # PyMuPDF
import json
import httpx
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

    def clean_page_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned_lines = []
        previous_line = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Boilerplate: exact match
            if re.match(r'^(Go to Index|Continuation Sheet)$', line, re.IGNORECASE):
                continue
            # Boilerplate: partial match (e.g., "Go to Index 2 | P a g e Continuation Sheet")
            if re.search(r'\bGo to Index\b', line, re.IGNORECASE):
                continue
            if re.search(r'\bContinuation Sheet\b', line, re.IGNORECASE):
                continue
            if re.match(r'^P\s*a\s*g\s*e\b', line, re.IGNORECASE):
                continue
            if re.match(r'^Page\s*\d+(\s*of\s*\d+)?$', line, re.IGNORECASE):
                continue
            # Lines like "2 | P a g e"
            if re.match(r'^\d+\s*\|\s*P\s*a\s*g\s*e', line, re.IGNORECASE):
                continue
            if re.match(r'^[A-Z\s]+Page\s*\d+$', line, re.IGNORECASE):
                continue
            if re.match(r'^(www\.|https?://)', line, re.IGNORECASE):
                continue
            if re.search(r'\b(NATIONAL STOCK EXCHANGE OF INDIA LIMITED|NSE INDIA|NSE)\b', line, re.IGNORECASE):
                continue
            if re.search(r'\b(Copyright|©|All rights reserved|Confidential)\b', line, re.IGNORECASE):
                continue
            if re.search(r'\b(Phone|Tel|Fax|Email|Contact|Address):?', line, re.IGNORECASE):
                continue
            if line == previous_line:
                continue
            cleaned_lines.append(line)
            previous_line = line

        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r'https?://\S+', '', cleaned_text)
        cleaned_text = re.sub(r'www\.\S+', '', cleaned_text)
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
        return cleaned_text.strip()

    def convert_tables_to_text(self, text: str) -> str:
        """Normalize table-like text into plain textual rows for embedding."""
        if not text:
            return ""

        normalized_lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if '|' in line:
                columns = [col.strip() for col in re.split(r'\s*\|\s*', line) if col.strip()]
                if columns:
                    normalized_lines.append(' | '.join(columns))
                    continue

            # Convert repeated spacing to delimiter for tabular-like rows
            line = re.sub(r'\s{2,}', ' | ', line)
            normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def normalize_for_boilerplate(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text.lower()

    def is_logo_text(self, text: str) -> bool:
        if not text:
            return False
        patterns = [
            r'national stock exchange',
            r'nse india',
            r'www\.nseindia\.com',
            r'nse\.com',
            r'logo',
            r'signature',
            r'confidential',
            r'proprietary'
        ]
        normalized = text.lower()
        return any(re.search(pattern, normalized) for pattern in patterns)

    def is_table_block(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r'\|\s*\w+|\bSr\.?\s*No\b|\bAmount\b|\bRs\.\b|\d+\s{2,}\w+', text))

    def is_figure_block(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r'\bFigure\b|\bChart\b|\bGraph\b|\bDiagram\b', text, re.IGNORECASE))

    def extract_page_regions(self, page: fitz.Page) -> List[Dict[str, Any]]:
        blocks = []
        page_height = page.rect.height
        header_cutoff = page_height * 0.18
        footer_cutoff = page_height * 0.82

        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *rest = block
            block_type = rest[0] if rest else None
            cleaned = self.clean_page_text(text)
            if not cleaned:
                continue

            region = "body"
            if y0 <= header_cutoff:
                region = "header"
            elif y1 >= footer_cutoff:
                region = "footer"
            if self.is_logo_text(cleaned):
                region = "logo"
            elif self.is_table_block(cleaned):
                region = "table"
            elif self.is_figure_block(cleaned):
                region = "figure"

            blocks.append({
                "region": region,
                "text": cleaned,
                "bbox": (x0, y0, x1, y1)
            })

        return blocks

    def detect_repeated_boilerplate(self, page_regions: List[List[Dict[str, Any]]]) -> set:
        text_occurrences = {}
        for page_idx, regions in enumerate(page_regions):
            seen = set()
            for region in regions:
                if region["region"] not in {"header", "footer", "logo"}:
                    continue
                normalized = self.normalize_for_boilerplate(region["text"])
                if not normalized:
                    continue
                if normalized in seen:
                    continue
                seen.add(normalized)
                text_occurrences.setdefault(normalized, set()).add(page_idx)

        threshold = max(1, math.ceil(len(page_regions) * 0.8))
        boilerplate = {text for text, pages in text_occurrences.items() if len(pages) >= threshold}
        if boilerplate:
            logger.info(f"Detected {len(boilerplate)} repeated boilerplate blocks across pages.")
        return boilerplate

    def collect_relevant_page_text(
        self,
        page_regions: List[Dict[str, Any]],
        boilerplate_texts: set,
        classification: str,
        vision_summary: Optional[str]
    ) -> str:
        extracted = []
        removed_headers = 0
        removed_footers = 0
        removed_boilerplate = 0

        for region in page_regions:
            normalized = self.normalize_for_boilerplate(region["text"])
            if not normalized:
                continue
            if region["region"] in {"header", "footer", "logo"}:
                if region["region"] in {"header", "footer"}:
                    removed_headers += 1
                continue
            if normalized in boilerplate_texts:
                removed_boilerplate += 1
                continue
            if region["region"] in {"body", "table", "figure"}:
                extracted.append(region["text"])

        if removed_headers or removed_footers or removed_boilerplate:
            logger.info(
                f"Page cleanup removed header/footer/boilerplate: headers={removed_headers}, footers={removed_footers}, boilerplate={removed_boilerplate}"
            )

        cleaned = "\n\n".join(extracted).strip()
        if not cleaned and vision_summary:
            return vision_summary
        return cleaned

    def generate_document_title(self, first_page_regions: List[Dict[str, Any]], doc_id: uuid.UUID, title_hint: Optional[str] = None) -> str:
        candidates = []
        for region in first_page_regions:
            if region["region"] not in {"header", "body"}:
                continue
            text = region["text"].strip()
            if not text or len(text.split()) < 3:
                continue
            if self.is_logo_text(text):
                continue
            if re.search(r'\b(www\.|https?://|Page\s*\d+|©|Copyright)\b', text, re.IGNORECASE):
                continue
            line = text.splitlines()[0].strip()
            if len(line.split()) >= 3:
                candidates.append(re.sub(r'\s+', ' ', line))
            if len(candidates) >= 3:
                break

        if candidates:
            title = candidates[0]
            title = title[:200].strip()
            logger.info(f"Generated title from first page content: {title}")
            return title

        if title_hint and title_hint.strip():
            cleaned_hint = title_hint.strip()
            if not re.match(r'^[\w-]+\.pdf$', cleaned_hint, re.IGNORECASE):
                logger.info(f"Using provided title hint after first page title generation failed: {cleaned_hint}")
                return cleaned_hint

        fallback = f"Document_{str(doc_id)[:8]}"
        logger.info(f"Title generation failed; using fallback title: {fallback}")
        return fallback

    def is_header_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if re.match(r'^\d+(\.\d+)*$', stripped):
            return True
        if re.match(r'^[A-Z][A-Z\s]+$', stripped):
            return True
        if re.match(r'^[A-Z].*:$', stripped):
            return True
        return False

    def extract_section_blocks(self, text: str) -> List[Dict[str, str]]:
        blocks = []
        section = ""
        subsection = ""
        current = []

        for line in text.splitlines():
            if self.is_header_line(line):
                if current:
                    blocks.append({
                        "section": section or "General",
                        "subsection": subsection or "",
                        "text": "\n".join(current).strip()
                    })
                    current = []

                stripped = line.strip()
                if re.match(r'^\d+(\.\d+)*$', stripped):
                    if stripped.count(".") == 0:
                        section = stripped
                        subsection = ""
                    else:
                        subsection = stripped
                        if not section:
                            section = stripped.split(".")[0]
                elif re.match(r'^[A-Z][A-Z\s]+$', stripped) or re.match(r'^[A-Z].*:$', stripped):
                    if not section:
                        section = stripped
                        subsection = ""
                    elif not subsection:
                        subsection = stripped
                    else:
                        subsection = stripped
                continue

            current.append(line)

        if current:
            blocks.append({
                "section": section or "General",
                "subsection": subsection or "",
                "text": "\n".join(current).strip()
            })

        return blocks

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Chunks page text into overlapping semantic windows using RecursiveCharacterTextSplitter."""
        if not text:
            return []

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        splits = splitter.split_text(text)
        normalized = []
        for chunk in splits:
            # Skip chunks that are too short to be meaningful (< 30 chars or < 5 words)
            if len(chunk.strip()) < 30 or len(chunk.split()) < 5:
                continue
            # Skip chunks that are pure table-of-contents boilerplate
            chunk_lower = chunk.lower()
            if 'go to index' in chunk_lower or 'continuation sheet' in chunk_lower:
                continue
            if normalized and len(chunk.split()) < 100:
                normalized[-1] = f"{normalized[-1]}\n{chunk}"
            else:
                normalized.append(chunk)
        return normalized

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
        # Title hint can be provided, but the final document title is generated from first page content.
        title_hint = name.strip() if name and name.strip() else None
        
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

        title_hint = name.strip() if name and name.strip() else None

        # 4. Open PDF and extract pages so the document title can be generated from content
        doc_fitz = fitz.open(file_path)
        num_pages = len(doc_fitz)
        if num_pages == 0:
            raise ValueError("PDF contains no pages.")

        # Collect page regions for repeated boilerplate detection and title generation
        page_regions_by_page = [self.extract_page_regions(doc_fitz[page_idx]) for page_idx in range(num_pages)]
        boilerplate_texts = self.detect_repeated_boilerplate(page_regions_by_page)
        doc_id = uuid.uuid4()
        title = self.generate_document_title(page_regions_by_page[0], doc_id, title_hint=title_hint)

        # Extract first 2 pages text for metadata generation
        first_pages_text = ""
        for page_idx in range(min(num_pages, 2)):
            first_pages_text += doc_fitz[page_idx].get_text() + "\n"

        # Create Document Entry using content-derived title
        doc = Document(
            id=doc_id,
            name=title,
            circular_number=circular_number,
            hash=file_hash,
            version=version,
            status="processing",
            issue_date=issue_date or date.today(),
            department=department or "Unknown",
            tags=tags or [],
            keywords=[],
            topics=[],
            entities=[],
            financial_terms=[],
            circular_type=""
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        doc_id_str = str(doc.id)
        await self.log_step(db, doc.id, "UPLOAD", "SUCCESS", f"Created document metadata. Title: {title}. Version: {version}")

        # Dynamic Metadata Extraction via LLM Qwen
        await self.log_step(db, doc.id, "METADATA_EXTRACTION", "RUNNING", "Running dynamic keyword/topic/entity extraction.")
        try:
            extracted_meta = await self.extract_document_metadata(first_pages_text)
            doc.keywords = extracted_meta.get("keywords") or []
            doc.topics = extracted_meta.get("topics") or []
            doc.entities = extracted_meta.get("entities") or []
            doc.financial_terms = extracted_meta.get("financial_terms") or []
            doc.circular_type = extracted_meta.get("circular_type") or "Circular"
            await db.commit()
            await db.refresh(doc)
            await self.log_step(db, doc.id, "METADATA_EXTRACTION", "SUCCESS", f"Metadata extracted: {json.dumps(extracted_meta)}")
        except Exception as meta_err:
            logger.error(f"Failed to save dynamic metadata: {meta_err}")
            await self.log_step(db, doc.id, "METADATA_EXTRACTION", "FAILED", f"Metadata extraction failed: {meta_err}")

        # Ensure index setup in Elasticsearch
        await search_store.init_indices()

        try:
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

                # 5c. Build page content from body/table/figure regions, removing headers and repeated boilerplate
                page_regions = page_regions_by_page[page_idx]
                content_text = self.collect_relevant_page_text(page_regions, boilerplate_texts, classification, vision_summary)
                cleaned_text = self.convert_tables_to_text(self.clean_page_text(content_text))

                page_blocks = self.extract_section_blocks(cleaned_text)
                if not page_blocks:
                    page_blocks = [{
                        "section": "General",
                        "subsection": "",
                        "text": cleaned_text or vision_summary or ""
                    }]

                if classification == "graph" and vision_summary:
                    page_blocks.append({
                        "section": "Graph",
                        "subsection": "",
                        "text": vision_summary
                    })

                qdrant_points = []
                for block_idx, block in enumerate(page_blocks):
                    block_text = block.get("text", "").strip()
                    if vision_summary and classification in ["scanned", "mixed"] and not block_text:
                        block_text = vision_summary

                    metadata_section = block.get("section") or "General"
                    metadata_subsection = block.get("subsection") or ""
                    metadata_type = classification if classification != "mixed" else "text"

                    if block_text:
                        parent_chunk_id = str(uuid.uuid5(doc.id, f"page_{page_num}_block_{block_idx}_section"))
                        parent_payload = {
                            "title": doc.name,
                            "doc_id": doc_id_str,
                            "page": page_num,
                            "section": metadata_section,
                            "subsection": metadata_subsection,
                            "parent_id": "",
                            "circular_number": circular_number or "",
                            "document_version": version or 1,
                            "type": "section",
                            "text": block_text,
                            "keywords": doc.keywords or [],
                            "topics": doc.topics or [],
                            "entities": doc.entities or [],
                            "financial_terms": doc.financial_terms or [],
                            "circular_type": doc.circular_type or ""
                        }
                        qdrant_points.append({
                            "id": parent_chunk_id,
                            "vector": None,
                            "payload": parent_payload
                        })
                        all_es_chunks.append({
                            "chunk_id": parent_chunk_id,
                            "doc_id": doc_id_str,
                            "title": doc.name,
                            "page_number": page_num,
                            "section": metadata_section,
                            "subsection": metadata_subsection,
                            "parent_id": "",
                            "type": "section",
                            "text": block_text,
                            "circular_number": circular_number or "",
                            "issue_date": str(doc.issue_date),
                            "version": version or 1,
                            "department": doc.department or "Unknown",
                            "tags": doc.tags or [],
                            "keywords": doc.keywords or [],
                            "topics": doc.topics or [],
                            "entities": doc.entities or [],
                            "financial_terms": doc.financial_terms or [],
                            "circular_type": doc.circular_type or "",
                            "status": doc.status or "active"
                        })

                    chunks = self.chunk_text(block_text)
                    if not chunks and block_text:
                        chunks = [block_text]

                    for chunk_idx, chunk_text in enumerate(chunks):
                        if not chunk_text.strip():
                            continue

                        chunk_id = str(uuid.uuid5(doc.id, f"page_{page_num}_block_{block_idx}_chunk_{chunk_idx}"))

                        payload = {
                            "title": doc.name,
                            "doc_id": doc_id_str,
                            "page": page_num,
                            "section": metadata_section,
                            "subsection": metadata_subsection,
                            "parent_id": parent_chunk_id,
                            "circular_number": circular_number or "",
                            "document_version": version or 1,
                            "type": metadata_type,
                            "text": chunk_text,
                            "keywords": doc.keywords or [],
                            "topics": doc.topics or [],
                            "entities": doc.entities or [],
                            "financial_terms": doc.financial_terms or [],
                            "circular_type": doc.circular_type or ""
                        }

                        qdrant_points.append({
                            "id": chunk_id,
                            "vector": None,
                            "payload": payload
                        })

                        all_es_chunks.append({
                            "chunk_id": chunk_id,
                            "doc_id": doc_id_str,
                            "title": doc.name,
                            "page_number": page_num,
                            "section": metadata_section,
                            "subsection": metadata_subsection,
                            "parent_id": parent_chunk_id,
                            "type": metadata_type,
                            "text": chunk_text,
                            "circular_number": circular_number or "",
                            "issue_date": str(doc.issue_date),
                            "version": version or 1,
                            "department": doc.department or "Unknown",
                            "tags": doc.tags or [],
                            "keywords": doc.keywords or [],
                            "topics": doc.topics or [],
                            "entities": doc.entities or [],
                            "financial_terms": doc.financial_terms or [],
                            "circular_type": doc.circular_type or "",
                            "status": doc.status or "active"
                        })

                        logger.debug(
                            f"Chunk created page={page_num} section={metadata_section} subsection={metadata_subsection} "
                            f"type={metadata_type} size={len(chunk_text)} preview={chunk_text[:120].replace('\n',' ')}"
                        )

                # Generate embeddings for the page-level chunks and upsert to Qdrant
                if qdrant_points:
                    vectors = await embedding_service.get_embeddings([p["payload"]["text"] for p in qdrant_points])
                    for idx, vector in enumerate(vectors):
                        qdrant_points[idx]["vector"] = vector
                    await qdrant_store.upsert_chunks(qdrant_points)

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

    async def extract_document_metadata(self, first_pages_text: str) -> Dict[str, Any]:
        """
        Uses Ollama reasoning model to dynamically extract metadata:
        Keywords, topics, entities, financial terms, and circular type.
        """
        prompt = (
            "You are a financial regulatory compliance analyzer. Analyze the following document introduction text and extract metadata.\n"
            "Produce structured JSON with the following keys:\n"
            "- keywords: A list of main keyword strings\n"
            "- topics: A list of key topics covered by this document\n"
            "- entities: A list of key organizations or regulatory bodies mentioned (e.g. SEBI, RBI, Exchange name, etc.)\n"
            "- financial_terms: A list of key financial or trading terms found (e.g. margin, derivatives, clearing, etc.)\n"
            "- circular_type: The type of circular or document (e.g., Circular, Guideline, Amendment, Notice, Policy, etc.)\n\n"
            f"Document Text:\n{first_pages_text[:6000]}\n\n"
            "Return ONLY a valid JSON object matching the keys."
        )

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{settings.llm.ollama_url}/api/chat",
                    json={
                        "model": settings.llm.reasoning_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a precise metadata extraction assistant. You only output valid JSON."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "stream": False,
                        "format": "json"
                    }
                )
                if response.status_code == 200:
                    content = response.json().get("message", {}).get("content", "{}")
                    parsed = json.loads(content)
                    logger.info(f"Successfully extracted document metadata: {parsed}")
                    return {
                        "keywords": parsed.get("keywords") or [],
                        "topics": parsed.get("topics") or [],
                        "entities": parsed.get("entities") or [],
                        "financial_terms": parsed.get("financial_terms") or [],
                        "circular_type": parsed.get("circular_type") or ""
                    }
                else:
                    logger.warning(f"Ollama metadata extraction API returned status {response.status_code}.")
        except Exception as e:
            logger.error(f"Ollama metadata extraction failed: {e}")
        
        # Fallback simulation
        return self._simulate_metadata_extraction(first_pages_text)

    def _simulate_metadata_extraction(self, text: str) -> Dict[str, Any]:
        """Self-healing simulated fallback for metadata extraction."""
        text_lower = text.lower()
        keywords = []
        topics = []
        entities = []
        financial_terms = []
        circular_type = "Circular"

        # Simple keyword heuristic
        if "margin" in text_lower:
            keywords.append("margin")
            financial_terms.append("margin collection")
        if "derivative" in text_lower:
            keywords.append("derivatives")
            topics.append("Currency Derivatives")
        if "sebi" in text_lower:
            entities.append("SEBI")
        if "nse" in text_lower:
            entities.append("NSE")
        if "clearing" in text_lower:
            financial_terms.append("clearing corporation")
        if "verification" in text_lower:
            keywords.append("verification")

        return {
            "keywords": keywords or ["compliance"],
            "topics": topics or ["regulatory compliance"],
            "entities": entities or ["Exchange"],
            "financial_terms": financial_terms or ["margin"],
            "circular_type": circular_type
        }

pdf_processor = PDFProcessor()
