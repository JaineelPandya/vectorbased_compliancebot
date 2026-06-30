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
from backend.app.services.title_extractor import extract_title_with_qwen2_5_0_5b
from backend.app.services.vision_pipeline import (
    vision_pipeline,
    CONTENT_TYPE_TEXT, CONTENT_TYPE_TABLE, CONTENT_TYPE_GRAPH,
    CONTENT_TYPE_FLOWCHART, CONTENT_TYPE_IMAGE, CONTENT_TYPE_MIXED,
    CONTENT_TYPE_UNKNOWN,
)

logger = logging.getLogger("app.pdf_processor")

# ---------------------------------------------------------------------------
# Helpers: Content-type mapping
# ---------------------------------------------------------------------------
_PAGE_CLASS_TO_CONTENT_TYPE: Dict[str, str] = {
    "text":      CONTENT_TYPE_TEXT,
    "table":     CONTENT_TYPE_TABLE,
    "graph":     CONTENT_TYPE_GRAPH,
    "scanned":   CONTENT_TYPE_IMAGE,
    "mixed":     CONTENT_TYPE_MIXED,
    "flowchart": CONTENT_TYPE_FLOWCHART,
    "unknown":   CONTENT_TYPE_UNKNOWN,
}


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

        # 2. Table Heuristic
        table_keywords = ["table", "annexure", "schedule", "statement of", "balance sheet", "particulars", "sr. no"]
        has_table_keywords = any(kw in text.lower() for kw in table_keywords)

        lines = text.split("\n")
        aligned_columns = 0
        for line in lines:
            if len(line.split("   ")) > 2 or "\t" in line:
                aligned_columns += 1

        if (aligned_columns > 3 or has_table_keywords) and drawing_count > 5:
            return "table", 0.85

        # 3. Graph Heuristic
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
            if re.match(r'^(Go to Index|Continuation Sheet)$', line, re.IGNORECASE):
                continue
            if re.search(r'\bGo to Index\b', line, re.IGNORECASE):
                continue
            if re.search(r'\bContinuation Sheet\b', line, re.IGNORECASE):
                continue
            if re.match(r'^P\s*a\s*g\s*e\b', line, re.IGNORECASE):
                continue
            if re.match(r'^Page\s*\d+(\s*of\s*\d+)?$', line, re.IGNORECASE):
                continue
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

    def _clean_title_candidate(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        cleaned = re.sub(r'^(Sub(?:ject)?|Re|Ref)\s*[:\-–]\s*', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'^[Pp]age\s*\d+(\s*of\s*\d+)?\s*[-–:]?\s*', '', cleaned).strip()
        cleaned = re.sub(r'\b(www\.|https?://)\S+', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def _is_valid_title_candidate(self, text: str) -> bool:
        if not text:
            return False
        if len(text.split()) < 3:
            return False
        if len(text) < 12 or len(text) > 120:
            return False
        if self.is_logo_text(text):
            return False
        if re.search(r'\b(www\.|https?://|©|copyright|confidential|page\s*\d+)\b', text, re.IGNORECASE):
            return False
        if re.match(r'^[\w\- ]+\.pdf$', text, re.IGNORECASE):
            return False
        if '_' in text and len(text.split()) <= 4 and re.search(r'\d', text):
            return False
        if re.match(r'^[\dIVXLCM\s\-\.]+$', text, re.IGNORECASE):
            return False
        if re.match(r'^[A-Z\s]{3,}$', text) and len(text.split()) < 4:
            return False
        return True

    def _rank_title_candidate(self, text: str) -> int:
        score = len(text)
        if re.search(r'\b(circular|guideline|notice|order|trading|regulation|compliance|notification|subject|announcement|schedule)\b', text, re.IGNORECASE):
            score += 40
        if re.search(r'\b(Sub|Subject|Re|Ref)\b', text, re.IGNORECASE):
            score -= 10
        if re.search(r'\b(NSE|SEBI|BSE)\b', text):
            score += 10
        return score

    def generate_document_title(
        self,
        first_page_regions: List[Dict[str, Any]],
        doc_id: uuid.UUID,
        first_pages_text: str = "",
        title_hint: Optional[str] = None,
        subject: Optional[str] = None
    ) -> str:
        candidates: List[str] = []

        # Add provided hint and subject as ranked candidates.
        if title_hint and title_hint.strip():
            hint = self._clean_title_candidate(title_hint.strip())
            if self._is_valid_title_candidate(hint):
                candidates.append(hint)
            else:
                logger.info(f"Ignoring weak title hint: {hint}")

        if subject and self._is_valid_title_candidate(subject):
            candidates.append(subject.strip())

        # Collect strong candidates from first page regions.
        for region in first_page_regions:
            if region["region"] not in {"header", "body"}:
                continue
            for line in region["text"].splitlines()[:8]:
                candidate = self._clean_title_candidate(line)
                if self._is_valid_title_candidate(candidate):
                    candidates.append(candidate)
            if len(candidates) >= 6:
                break

        # If no candidate from regions, look through first pages raw text.
        if not candidates and first_pages_text:
            for line in first_pages_text.splitlines()[:20]:
                candidate = self._clean_title_candidate(line)
                if self._is_valid_title_candidate(candidate):
                    candidates.append(candidate)
                if len(candidates) >= 4:
                    break

        if candidates:
            candidates = sorted(set(candidates), key=self._rank_title_candidate, reverse=True)
            title = candidates[0][:200].strip()
            logger.info(f"Generated title from title/subject/first-page content: {title}")
            return title

        if subject and self._is_valid_title_candidate(subject):
            logger.info(f"Using subject line as fallback document title: {subject}")
            return subject[:200].strip()

        fallback = f"Document_{str(doc_id)[:8]}"
        logger.info(f"Title generation failed; using fallback title: {fallback}")
        return fallback

    # ------------------------------------------------------------------
    # Fix 3: Extract Subject field from first-page text
    # ------------------------------------------------------------------
    def extract_document_subject(self, first_page_text: str) -> str:
        """
        Extracts the subject line from regulatory circular text.
        Looks for patterns like:
          Sub: Trading hours for commodity derivatives segment
          Subject: Trading hours for commodity derivatives segment
        Returns empty string if not found.
        """
        patterns = [
            r'(?:Sub(?:ject)?)\s*[:\-–]\s*(.+?)(?:\n|$)',
            r'(?:Re|Ref)\s*[:\-–]\s*(.+?)(?:\n|$)',
        ]
        for pat in patterns:
            m = re.search(pat, first_page_text, re.IGNORECASE)
            if m:
                subject = m.group(1).strip()
                subject = re.sub(r'\s+', ' ', subject)
                logger.info(f"Extracted document subject: {subject[:120]}")
                return subject[:500]
        return ""

    # ------------------------------------------------------------------
    # Fix 1: Hierarchical semantic chunking
    # ------------------------------------------------------------------
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

    def _is_subject_line(self, line: str) -> bool:
        """Detect subject / sub / re lines — never split these."""
        return bool(re.match(r'^\s*(?:Sub(?:ject)?|Re|Ref)\s*[:\-–]', line, re.IGNORECASE))

    def _is_section_heading(self, line: str) -> bool:
        """
        Hierarchical section detection for NSE/SEBI/BSE circular format:
          - Numbered: 1. 2. 3. / 1.1 / 1.1.1
          - ALL-CAPS heading
          - Title-Case followed by colon
        """
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return False
        # Numbered section
        if re.match(r'^\d+(\.\d+)*[\.\)]\s+\S', stripped):
            return True
        # ALL CAPS heading (min 3 words)
        if re.match(r'^[A-Z][A-Z\s\-&/]{8,}$', stripped):
            return True
        # Title colon pattern
        if re.match(r'^[A-Z][a-zA-Z\s]{3,}:\s*$', stripped):
            return True
        return False

    def hierarchical_chunk_document(
        self,
        text: str,
        title: str,
        subject: str,
        doc_type: str = "TEXT"
    ) -> List[Dict[str, str]]:
        """
        Hierarchical chunking that respects document structure.

        Chunking strategy by content_type:
          ┌─────────────────────────────────────────────────────────────┐
          │ TEXT          → Hierarchical sections → Sliding window    │
          │ TABLE         → Hierarchical sections → Full-row intact   │
          │ GRAPH/IMAGE   → Single JSON chunk (via ingest_pdf)         │
          └─────────────────────────────────────────────────────────────┘

        TITLE and SUBJECT chunks are always fixed (no windowing — must not be fragmented).
        """
        chunks: List[Dict[str, str]] = []

        # --- Always create a TITLE chunk (importance_score=1.0) ---
        if title:
            chunks.append({
                "section":          "Title",
                "text":             title,
                "type":             doc_type,
                "importance_score": "1.0"
            })

        # --- Always create a SUBJECT chunk (importance_score=1.0) ---
        if subject:
            chunks.append({
                "section":          "Subject",
                "text":             f"Subject: {subject}",
                "type":             doc_type,
                "importance_score": "1.0"
            })

        if not text:
            return chunks

        # Decide body splitting strategy based on content_type
        use_sliding_window = doc_type in (CONTENT_TYPE_TEXT, CONTENT_TYPE_UNKNOWN, "TEXT", "UNKNOWN", "")

        current_section = "General"
        current_lines: List[str] = []

        def flush_section(section_name: str, lines: List[str]) -> None:
            body = "\n".join(lines).strip()
            if not body:
                return

            is_important = (
                "sub:" in body.lower()[:30]
                or "subject:" in body.lower()[:30]
                or section_name in ("Title", "Subject")
            )

            if use_sliding_window:
                # ------------------------------------------------
                # TEXT: Sentence-aware sliding window
                # ------------------------------------------------
                windows = self._sliding_window_text_chunks(
                    body,
                    chunk_size=settings.chunking.text_chunk_size,
                    overlap_sentences=settings.chunking.text_overlap_sentences
                )
                for win in windows:
                    if len(win.strip()) < settings.chunking.text_min_chunk_len:
                        continue
                    chunks.append({
                        "section":          section_name,
                        "text":             win.strip(),
                        "type":             doc_type,
                        "importance_score": "1.0" if is_important else "0.0"
                    })
            else:
                # ------------------------------------------------
                # TABLE: Keep full rows intact, split only if
                # extremely long (avoids cutting table rows mid-way)
                # ------------------------------------------------
                if len(body) > 1500:
                    sub_chunks = self._split_table_rows(body, max_rows=settings.chunking.table_max_rows)
                else:
                    sub_chunks = [body]
                for sc in sub_chunks:
                    if len(sc.strip()) < settings.chunking.text_min_chunk_len:
                        continue
                    chunks.append({
                        "section":          section_name,
                        "text":             sc.strip(),
                        "type":             doc_type,
                        "importance_score": "1.0" if is_important else "0.0"
                    })

        for raw_line in text.splitlines():
            line = raw_line.rstrip()

            # Preserve subject lines verbatim in their own section
            if self._is_subject_line(line) and not current_lines:
                flush_section(current_section, current_lines)
                current_lines = []
                current_section = "Subject"
                current_lines.append(line)
                continue

            if self._is_section_heading(line):
                flush_section(current_section, current_lines)
                current_section = line.strip()
                current_lines   = []
                continue

            current_lines.append(line)

        # Flush remaining
        flush_section(current_section, current_lines)

        return chunks

    # ------------------------------------------------------------------
    # Sentence-aware sliding window chunker (TEXT pages only)
    # ------------------------------------------------------------------
    def _sliding_window_text_chunks(
        self,
        text: str,
        chunk_size: int = 600,
        overlap_sentences: int = 2
    ) -> List[str]:
        """
        Sliding window chunking on sentence boundaries.

        Algorithm:
          1. Split body into sentences (split on [.!?] boundaries).
          2. Accumulate sentences into a window until `chunk_size` chars is reached.
          3. Slide forward: the next window starts `overlap_sentences` sentences
             before where the current window ended, so adjacent windows share
             context and no information is lost at boundaries.
          4. If the entire body fits in one window, return it as-is (no fragmentation).

        Args:
          text:              Section body text to split.
          chunk_size:        Max characters per window (default 600 ≈ 120 words).
          overlap_sentences: Number of sentences to carry into the next window.

        Returns:
          List of overlapping text windows, each ≤ chunk_size characters.
        """
        # Split into sentences preserving the delimiter
        raw_sentences = re.split(r'(?<=[.!?\u0964])\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return [text.strip()] if text.strip() else []

        # If body fits entirely in one window, no splitting needed
        if len(text) <= chunk_size:
            return [text.strip()]

        windows: List[str] = []
        i = 0

        while i < len(sentences):
            # Build current window
            window_sents: List[str] = []
            current_len = 0
            j = i

            while j < len(sentences):
                s = sentences[j]
                # Always include at least one sentence even if it exceeds chunk_size
                if current_len + len(s) + 1 > chunk_size and window_sents:
                    break
                window_sents.append(s)
                current_len += len(s) + 1
                j += 1

            window_text = " ".join(window_sents)
            if window_text.strip():
                windows.append(window_text.strip())

            # Reached the end of sentences
            if j >= len(sentences):
                break

            # Slide forward: step back by overlap_sentences from j
            # so next window begins overlap_sentences before the cut point
            next_i = max(i + 1, j - overlap_sentences)
            i = next_i

        return windows if windows else [text.strip()]

    # ------------------------------------------------------------------
    # Table-row-aware splitter (TABLE pages only)
    # ------------------------------------------------------------------
    def _split_table_rows(self, text: str, max_rows: int = 20) -> List[str]:
        """
        Split table text by rows (newlines), keeping up to `max_rows` per chunk.
        Preserves header row in every chunk for context.
        """
        rows = [r for r in text.splitlines() if r.strip()]
        if len(rows) <= max_rows:
            return [text]

        # Treat first row as header if it looks like one
        header_row = rows[0] if rows else ""
        data_rows  = rows[1:] if len(rows) > 1 else rows

        result = []
        for start in range(0, len(data_rows), max_rows):
            batch = data_rows[start:start + max_rows]
            # Re-prepend header so each chunk is self-contained
            chunk_text = "\n".join([header_row] + batch)
            result.append(chunk_text)

        return result if result else [text]

    def _split_long_section(self, text: str, max_chars: int = 900, overlap: int = 80) -> List[str]:
        """Legacy sentence-boundary splitter — kept for backward compatibility."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result    = []
        current   = ""
        prev_tail = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                result.append(prev_tail + current)
                words      = current.split()
                prev_tail  = " ".join(words[-overlap // 10:]) + " " if words else ""
                current    = sentence
            else:
                current += (" " if current else "") + sentence

        if current:
            result.append(prev_tail + current)

        return result

    # ------------------------------------------------------------------
    # Fix 2: Subject-boosted embedding text
    # ------------------------------------------------------------------
    def build_boosted_text(self, title: str, subject: str, section: str, text: str) -> str:
        """
        Prepend document context before embedding.
        The original `text` is stored in the payload;
        this string is used ONLY for generating the vector.
        """
        parts = []
        if title:
            parts.append(f"TITLE:\n{title}")
        if subject:
            parts.append(f"SUBJECT:\n{subject}")
        if section and section not in ("General", ""):
            parts.append(f"SECTION:\n{section}")
        parts.append(text)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Legacy method kept for compatibility (used by extract_section_blocks callers)
    # ------------------------------------------------------------------
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
        """Legacy fixed-window chunker — kept for fallback compatibility."""
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
            if len(chunk.strip()) < 30 or len(chunk.split()) < 5:
                continue
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
        Uses hierarchical chunking (Fix 1), subject-boosted embeddings (Fix 2),
        and content_type / vision_confidence payload fields (Fix 10).
        """
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
            stmt_circ = select(Document).where(Document.circular_number == circular_number)
            res_circ = await db.execute(stmt_circ)
            matched_docs = res_circ.scalars().all()

            if matched_docs:
                max_version = max(d.version for d in matched_docs)
                version = max_version + 1

                await db.execute(
                    update(Document)
                    .where(Document.circular_number == circular_number)
                    .values(status="archived")
                )
                await db.commit()

        title_hint = name.strip() if name and name.strip() else None

        # 4. Open PDF
        doc_fitz = fitz.open(file_path)
        num_pages = len(doc_fitz)
        if num_pages == 0:
            raise ValueError("PDF contains no pages.")

        page_regions_by_page = [self.extract_page_regions(doc_fitz[page_idx]) for page_idx in range(num_pages)]
        boilerplate_texts = self.detect_repeated_boilerplate(page_regions_by_page)
        doc_id = uuid.uuid4()

        # Extract first 2 pages text for subject + metadata extraction early.
        first_pages_text = ""
        for page_idx in range(min(num_pages, 2)):
            first_pages_text += doc_fitz[page_idx].get_text() + "\n"

        # Fix 3: Extract subject from first-page text before title generation,
        # so the subject line can also serve as a strong fallback title.
        subject = self.extract_document_subject(first_pages_text)
        title = self.generate_document_title(
            page_regions_by_page[0],
            doc_id,
            first_pages_text=first_pages_text,
            title_hint=title_hint,
            subject=subject
        )

        # LLM Title Extraction Fallback for cases where page heuristics are too weak.
        if title.startswith("Document_") or re.match(r'^[\w-]+\.pdf$', title, re.IGNORECASE):
            extracted_title = await extract_title_with_qwen2_5_0_5b(first_pages_text)
            if extracted_title:
                title = extracted_title
                logger.info(f"Replaced fallback title with LLM-extracted title: {title}")

        # Create Document Entry
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
            circular_type="",
            subject=subject   # P0: persist subject to DB
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        doc_id_str = str(doc.id)
        await self.log_step(db, doc.id, "UPLOAD", "SUCCESS", f"Created document metadata. Title: {title}. Subject: {subject}. Version: {version}")

        # P2: Permanently copy the PDF to uploads dir so reindex can locate it by hash
        import shutil
        upload_dest = os.path.join(
            settings.storage.upload_dir,
            f"{doc_id_str}_{os.path.basename(file_path)}"
        )
        try:
            shutil.copy2(file_path, upload_dest)
            logger.info(f"PDF permanently saved to uploads dir: {upload_dest}")
        except Exception as copy_err:
            logger.warning(f"Could not copy PDF to uploads dir: {copy_err}. Reindex will require re-upload.")

        # Dynamic Metadata Extraction via LLM
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

        # Counters for reindex response
        total_chunks_created = 0
        table_chunks_created = 0
        graph_chunks_created = 0

        try:
            await self.log_step(db, doc.id, "PARSING", "SUCCESS", f"Opened PDF with {num_pages} pages.")
            all_es_chunks = []

            for page_idx in range(num_pages):
                page = doc_fitz[page_idx]
                page_num = page_idx + 1

                # 5a. Classify Page (heuristic)
                classification, confidence = self.classify_page(page)

                vision_summary = None
                vision_extracted_values = None
                vision_confidence = 0.0
                content_type = _PAGE_CLASS_TO_CONTENT_TYPE.get(classification, CONTENT_TYPE_TEXT)

                # 5b. Vision pipeline — cascade approach
                # Pass fitz_page so Stage 0 heuristics can skip VL on text-only pages
                if classification in ["scanned", "table", "graph", "mixed"]:
                    await self.log_step(db, doc.id, "VISION", "RUNNING", f"Vision cascade for page {page_num} ({classification})")
                    pix = page.get_pixmap(dpi=150)
                    image_bytes = pix.tobytes("png")

                    vision_result = await vision_pipeline.analyze_page_image(
                        image_bytes=image_bytes,
                        page_type=classification,
                        page_num=page_num,
                        fitz_page=page
                    )

                    # Fix 10: use VL's classification as the authoritative content_type
                    vl_classification = vision_result.get("classification", classification)
                    content_type      = vision_result.get("content_type", content_type)
                    vision_confidence = float(vision_result.get("vision_confidence", 0.0))
                    confidence        = vision_result.get("confidence", confidence)

                    # Fix 4+6: only use summary from VL if it's NOT fabricated/unknown
                    if vl_classification not in ("unknown",) and not vision_result.get("needs_manual_review"):
                        vision_summary          = vision_result.get("summary") or None
                        vision_extracted_values = vision_result.get("extracted_values") or None
                    else:
                        vision_summary          = None
                        vision_extracted_values = None

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

                # 5c. Build page text
                page_regions  = page_regions_by_page[page_idx]
                content_text  = self.collect_relevant_page_text(page_regions, boilerplate_texts, classification, vision_summary)
                cleaned_text  = self.convert_tables_to_text(self.clean_page_text(content_text))

                # Fix 1: Hierarchical chunking
                # For graph chunks that passed threshold — create a graph chunk from extracted_values
                # For text/table/mixed — use hierarchical text chunking
                page_chunks: List[Dict[str, str]] = []

                # Fix 9: Only create GRAPH chunks when confidence is high enough
                if content_type == CONTENT_TYPE_GRAPH and vision_confidence > 0.8 and vision_extracted_values:
                    graph_summary = json.dumps(vision_extracted_values)
                    page_chunks.append({
                        "section":         "Graph",
                        "text":            graph_summary,
                        "type":            CONTENT_TYPE_GRAPH,
                        "importance_score": "0.0",
                        "vision_confidence": str(vision_confidence),
                        "content_type":    CONTENT_TYPE_GRAPH
                    })
                    graph_chunks_created += 1

                if content_type in (CONTENT_TYPE_TABLE, CONTENT_TYPE_MIXED) and cleaned_text:
                    # Table / mixed — hierarchical chunk the text
                    hier_chunks = self.hierarchical_chunk_document(
                        cleaned_text, title, subject, doc_type=content_type
                    )
                    page_chunks.extend(hier_chunks)
                    if content_type == CONTENT_TYPE_TABLE:
                        table_chunks_created += len(hier_chunks)

                elif content_type not in (CONTENT_TYPE_GRAPH,) and cleaned_text:
                    # Regular text page
                    hier_chunks = self.hierarchical_chunk_document(
                        cleaned_text, title, subject, doc_type=content_type
                    )
                    page_chunks.extend(hier_chunks)

                # Fallback: if no chunks but we have text or vision summary
                if not page_chunks:
                    fallback_text = cleaned_text or ""
                    if not fallback_text and vision_summary:
                        fallback_text = vision_summary
                    if fallback_text and len(fallback_text.strip()) >= 30:
                        page_chunks.append({
                            "section":         "General",
                            "text":            fallback_text.strip(),
                            "type":            content_type,
                            "importance_score": "0.0",
                            "vision_confidence": str(vision_confidence),
                            "content_type":    content_type
                        })

                # Build Qdrant and ES records for this page
                qdrant_points = []
                for chunk_idx, chunk in enumerate(page_chunks):
                    chunk_text_raw = chunk.get("text", "").strip()

                    # Fix 12: Skip low-quality chunks during ingestion
                    if len(chunk_text_raw) < 50:
                        logger.debug(f"Skipping short chunk (len={len(chunk_text_raw)}) page={page_num}")
                        continue
                    if "fallback extraction" in chunk_text_raw.lower():
                        logger.warning(f"Skipping fabricated fallback chunk on page={page_num}")
                        continue

                    chunk_section      = chunk.get("section", "General")
                    chunk_content_type = chunk.get("content_type", content_type)
                    chunk_vis_conf     = float(chunk.get("vision_confidence", vision_confidence))
                    chunk_importance   = float(chunk.get("importance_score", "0.0"))

                    # Fix 9: Skip low-confidence GRAPH chunks in ingestion
                    if chunk_content_type == CONTENT_TYPE_GRAPH and chunk_vis_conf < 0.8:
                        logger.info(f"Skipping GRAPH chunk with low confidence {chunk_vis_conf:.2f} on page={page_num}")
                        continue

                    chunk_id = str(uuid.uuid5(doc.id, f"page_{page_num}_hier_{chunk_idx}"))

                    # Fix 2: Build boosted text for embedding only
                    boosted_text = self.build_boosted_text(title, subject, chunk_section, chunk_text_raw)

                    payload = {
                        "title":             doc.name,
                        "subject":           subject,
                        "doc_id":            doc_id_str,
                        "page":              page_num,
                        "section":           chunk_section,
                        "subsection":        "",
                        "parent_id":         "",
                        "circular_number":   circular_number or "",
                        "document_version":  version or 1,
                        # Fix 10: content_type replaces generic "type"
                        "type":              chunk_content_type,
                        "content_type":      chunk_content_type,
                        "text":              chunk_text_raw,       # original text stored
                        "vision_confidence": chunk_vis_conf,       # Fix 9
                        "importance_score":  chunk_importance,     # Fix 13
                        "keywords":          doc.keywords or [],
                        "topics":            doc.topics or [],
                        "entities":          doc.entities or [],
                        "financial_terms":   doc.financial_terms or [],
                        "circular_type":     doc.circular_type or ""
                    }

                    qdrant_points.append({
                        "id":      chunk_id,
                        "vector":  None,   # filled below
                        "payload": payload,
                        "_boosted_text": boosted_text   # for embedding only
                    })

                    all_es_chunks.append({
                        "chunk_id":         chunk_id,
                        "doc_id":           doc_id_str,
                        "title":            doc.name,
                        "subject":          subject,
                        "page_number":      page_num,
                        "section":          chunk_section,
                        "subsection":       "",
                        "parent_id":        "",
                        "type":             chunk_content_type,
                        "content_type":     chunk_content_type,
                        "text":             chunk_text_raw,
                        "circular_number":  circular_number or "",
                        "issue_date":       str(doc.issue_date),
                        "version":          version or 1,
                        "department":       doc.department or "Unknown",
                        "tags":             doc.tags or [],
                        "keywords":         doc.keywords or [],
                        "topics":           doc.topics or [],
                        "entities":         doc.entities or [],
                        "financial_terms":  doc.financial_terms or [],
                        "circular_type":    doc.circular_type or "",
                        "status":           doc.status or "active",
                        "vision_confidence": chunk_vis_conf,
                        "importance_score": chunk_importance
                    })

                    total_chunks_created += 1
                    logger.debug(
                        f"Chunk created page={page_num} section={chunk_section} "
                        f"content_type={chunk_content_type} importance={chunk_importance} "
                        f"vis_conf={chunk_vis_conf:.2f} size={len(chunk_text_raw)} "
                        f"preview={chunk_text_raw[:120].replace(chr(10), ' ')}"
                    )

                # Fix 2: Generate embeddings using BOOSTED text
                if qdrant_points:
                    boosted_texts = [p["_boosted_text"] for p in qdrant_points]
                    vectors = await embedding_service.get_embeddings(boosted_texts)
                    for idx, vector in enumerate(vectors):
                        qdrant_points[idx]["vector"] = vector
                        del qdrant_points[idx]["_boosted_text"]   # remove temp key
                    await qdrant_store.upsert_chunks(qdrant_points)

            # 6. Index Document Metadata in Elasticsearch
            es_metadata = {
                "name":            doc.name,
                "circular_number": doc.circular_number,
                "issue_date":      str(doc.issue_date),
                "version":         doc.version,
                "department":      doc.department,
                "tags":            doc.tags,
                "status":          "active",
                "uploaded_at":     doc.uploaded_at.isoformat()
            }
            await search_store.index_document_metadata(doc_id_str, es_metadata)

            if all_es_chunks:
                await search_store.index_chunks(all_es_chunks)

            # 7. Update Document Status
            doc.status = "active"
            await db.commit()
            await self.log_step(
                db, doc.id, "COMPLETE", "SUCCESS",
                f"Document fully processed. total_chunks={total_chunks_created} "
                f"table_chunks={table_chunks_created} graph_chunks={graph_chunks_created}"
            )
            doc._ingest_stats = {
                "chunks_created": total_chunks_created,
                "table_chunks":   table_chunks_created,
                "graph_chunks":   graph_chunks_created
            }
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
                        "keywords":       parsed.get("keywords") or [],
                        "topics":         parsed.get("topics") or [],
                        "entities":       parsed.get("entities") or [],
                        "financial_terms": parsed.get("financial_terms") or [],
                        "circular_type":  parsed.get("circular_type") or ""
                    }
                else:
                    logger.warning(f"Ollama metadata extraction API returned status {response.status_code}.")
        except Exception as e:
            logger.error(f"Ollama metadata extraction failed: {e}")

        return self._simulate_metadata_extraction(first_pages_text)

    def _simulate_metadata_extraction(self, text: str) -> Dict[str, Any]:
        """Self-healing simulated fallback for metadata extraction."""
        text_lower = text.lower()
        keywords = []
        topics = []
        entities = []
        financial_terms = []
        circular_type = "Circular"

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
            "keywords":       keywords or ["compliance"],
            "topics":         topics or ["regulatory compliance"],
            "entities":       entities or ["Exchange"],
            "financial_terms": financial_terms or ["margin"],
            "circular_type":  circular_type
        }


pdf_processor = PDFProcessor()
