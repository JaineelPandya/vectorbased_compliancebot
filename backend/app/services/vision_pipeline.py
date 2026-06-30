import logging
import httpx
import base64
import json
from typing import Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
from backend.app.config import settings

logger = logging.getLogger("app.vision_pipeline")

# ---------------------------------------------------------------------------
# Content-type constants (Fix 10)
# ---------------------------------------------------------------------------
CONTENT_TYPE_TEXT      = "TEXT"
CONTENT_TYPE_TABLE     = "TABLE"
CONTENT_TYPE_GRAPH     = "GRAPH"
CONTENT_TYPE_FLOWCHART = "FLOWCHART"
CONTENT_TYPE_IMAGE     = "IMAGE"
CONTENT_TYPE_MIXED     = "MIXED"
CONTENT_TYPE_UNKNOWN   = "UNKNOWN"

# Map VL classification labels → canonical content_type
_VL_TO_CONTENT_TYPE: Dict[str, str] = {
    "text_document": CONTENT_TYPE_TEXT,
    "table":         CONTENT_TYPE_TABLE,
    "chart_graph":   CONTENT_TYPE_GRAPH,
    "flowchart":     CONTENT_TYPE_FLOWCHART,
    "image":         CONTENT_TYPE_IMAGE,
    "mixed":         CONTENT_TYPE_MIXED,
    "unknown":       CONTENT_TYPE_UNKNOWN,
}

GRAPH_CONFIDENCE_THRESHOLD = 0.85   # Fix 6


class VisionPipeline:
    def __init__(self):
        self.model = settings.llm.vision_model
        self.url   = settings.llm.ollama_url

    # ------------------------------------------------------------------
    # Stage 0 — Cheap PyMuPDF heuristics (Fix 5 cascade)
    # ------------------------------------------------------------------
    def _stage0_heuristics(self, page: fitz.Page) -> Tuple[bool, str]:
        """
        Returns (needs_vl_call, hint_classification).
        If the page is clearly text-only we skip all VL calls entirely.
        """
        text        = page.get_text().strip()
        images      = page.get_images()
        drawings    = page.get_drawings()
        char_count  = len(text)
        image_count = len(images)
        draw_count  = len(drawings)

        # Purely text page — no VL needed
        if char_count >= 200 and image_count == 0 and draw_count <= 5:
            return False, "text_document"

        # Scanned / image-heavy page → needs VL
        if char_count < 100 and image_count >= 1:
            return True, "image"

        # High drawing density with little text → likely chart
        if draw_count > 30 and char_count < 300:
            return True, "chart_graph"

        # Moderate complexity → run VL classification
        if image_count >= 1 or draw_count > 10:
            return True, "mixed"

        # Default: mostly text
        return False, "text_document"

    # ------------------------------------------------------------------
    # Stage 1 — VL classification call (Fix 5)
    # ------------------------------------------------------------------
    async def _stage1_classify(
        self,
        base64_image: str,
        page_num: int,
        hint: str
    ) -> Dict[str, Any]:
        """
        Calls Qwen-VL with a lightweight classification prompt.
        Returns {"classification": str, "confidence": float}.
        On failure returns unknown with confidence 0.0 — never fabricates.
        """
        system_prompt = (
            "You are a precise document page classifier. "
            "Classify the page into EXACTLY ONE of the following categories:\n"
            "  text_document  — page is primarily readable text (paragraphs, headings, bullets)\n"
            "  table          — structured grid with rows and columns or bordered cells\n"
            "  chart_graph    — bar chart, line chart, pie chart, candlestick, scatter, or any data-driven chart\n"
            "  flowchart      — organisation chart, process diagram, decision tree\n"
            "  image          — photograph, logo, scanned image with no structured data\n"
            "  mixed          — combination of text + table OR text + logo (most regulatory circulars)\n"
            "  unknown        — cannot be determined\n\n"
            "IMPORTANT RULES:\n"
            "  - Coloured text or coloured headings ≠ graph\n"
            "  - A logo next to text = mixed, NOT graph\n"
            "  - Tables with visible borders = table\n"
            "  - Bar charts / line charts / pie charts / candlesticks = chart_graph\n"
            "  - Organisation diagrams with boxes and arrows = flowchart\n\n"
            "Return ONLY valid JSON with exactly two keys:\n"
            '{"classification": "<category>", "confidence": <float 0.0-1.0>}'
        )
        user_prompt = (
            f"Classify this document page (page {page_num}). "
            f"Hint from heuristics: '{hint}'. "
            "Return JSON only."
        )

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(
                    f"{self.url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": user_prompt,
                                "images": [base64_image]
                            }
                        ],
                        "stream": False,
                        "format": "json"
                    }
                )
                if response.status_code == 200:
                    content = response.json().get("message", {}).get("content", "{}")
                    parsed  = json.loads(content)
                    cls     = parsed.get("classification", "unknown").lower().strip()
                    conf    = float(parsed.get("confidence", 0.0))
                    if cls not in _VL_TO_CONTENT_TYPE:
                        cls = "unknown"
                    logger.info(
                        f"[Vision Stage1] page={page_num} classification={cls} confidence={conf:.2f}"
                    )
                    return {"classification": cls, "confidence": conf}
                else:
                    logger.warning(
                        f"[Vision Stage1] VL API returned status {response.status_code} for page {page_num}"
                    )
        except Exception as e:
            logger.warning(f"[Vision Stage1] VL classification failed for page {page_num}: {e}")

        # Fix 4: never fabricate — return unknown safely
        return {"classification": "unknown", "confidence": 0.0}

    # ------------------------------------------------------------------
    # Stage 2 — Deep structured extraction (Fix 8)
    # ------------------------------------------------------------------
    async def _stage2_extract(
        self,
        base64_image: str,
        classification: str,
        page_num: int
    ) -> Dict[str, Any]:
        """
        Deep extraction only for high-confidence chart_graph / table / flowchart.
        Returns structured payload — never free-text fabrication.
        """
        if classification == "chart_graph":
            system_prompt = (
                "You are a precise data extraction assistant for financial charts. "
                "Extract structured chart data from the image. "
                "Return ONLY valid JSON with these keys:\n"
                '{"graph_type": "line_chart|bar_chart|pie_chart|candlestick|scatter|other", '
                '"title": "", "x_axis": "", "y_axis": "", "series": []}'
            )
            user_prompt = (
                f"Extract structured data from this chart on page {page_num}. "
                "graph_type must be one of: line_chart, bar_chart, pie_chart, candlestick, scatter, other. "
                "series is a list of {label, values[]} objects. "
                "Return JSON only."
            )
        elif classification == "table":
            system_prompt = (
                "You are a precise table extraction assistant for regulatory documents. "
                "Extract all rows and columns from the table image. "
                "Return ONLY valid JSON:\n"
                '{"table_title": "", "headers": [], "rows": [{"col1": "val1", ...}]}'
            )
            user_prompt = (
                f"Extract the full table from this page {page_num}. "
                "Include all rows. Return JSON only."
            )
        elif classification == "flowchart":
            system_prompt = (
                "You are a diagram extraction assistant. "
                "Extract the structure of this flowchart or organisation chart. "
                "Return ONLY valid JSON:\n"
                '{"diagram_title": "", "nodes": [], "edges": []}'
            )
            user_prompt = (
                f"Extract the flowchart or org-chart structure from page {page_num}. "
                "Return JSON only."
            )
        else:
            # Should not reach here, but guard anyway
            return {"extracted_values": {}, "summary": ""}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": user_prompt,
                                "images": [base64_image]
                            }
                        ],
                        "stream": False,
                        "format": "json"
                    }
                )
                if response.status_code == 200:
                    content = response.json().get("message", {}).get("content", "{}")
                    parsed  = json.loads(content)
                    logger.info(
                        f"[Vision Stage2] page={page_num} classification={classification} "
                        f"extracted keys={list(parsed.keys())}"
                    )
                    return {"extracted_values": parsed, "summary": ""}
                else:
                    logger.warning(
                        f"[Vision Stage2] VL extraction returned status {response.status_code} "
                        f"for page {page_num}"
                    )
        except Exception as e:
            logger.warning(f"[Vision Stage2] Deep extraction failed for page {page_num}: {e}")

        # Fix 4: on failure return empty — never fabricate
        return {"extracted_values": {}, "summary": ""}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def analyze_page_image(
        self,
        image_bytes: bytes,
        page_type: str,
        page_num: int,
        fitz_page: Optional[fitz.Page] = None
    ) -> Dict[str, Any]:
        """
        Three-stage cascade (Fix 5):
          Stage 0: PyMuPDF heuristics — skip VL for text-only pages
          Stage 1: VL classification — lightweight single call
          Stage 2: VL deep extraction — only for high-confidence visual pages

        Returns a dict with:
          type, content_type, classification, confidence,
          summary, extracted_values, needs_manual_review
        """
        # ---- Stage 0 ----
        if fitz_page is not None:
            needs_vl, hint = self._stage0_heuristics(fitz_page)
        else:
            # No fitz page provided — decide from page_type hint
            needs_vl = page_type not in ("text",)
            hint = page_type

        if not needs_vl:
            logger.info(
                f"[Vision Stage0] page={page_num} classified as text by heuristics — skipping VL calls."
            )
            return {
                "type":               "text_document",
                "content_type":       CONTENT_TYPE_TEXT,
                "classification":     "text_document",
                "confidence":         1.0,
                "summary":            "",
                "extracted_values":   {},
                "needs_manual_review": False,
                "vision_confidence":  1.0
            }

        # ---- Stage 1 ----
        base64_image   = base64.b64encode(image_bytes).decode("utf-8")
        stage1_result  = await self._stage1_classify(base64_image, page_num, hint)
        classification = stage1_result["classification"]
        confidence     = stage1_result["confidence"]
        content_type   = _VL_TO_CONTENT_TYPE.get(classification, CONTENT_TYPE_UNKNOWN)

        # Fix 6: Downgrade low-confidence chart_graph to unknown
        if classification == "chart_graph" and confidence <= GRAPH_CONFIDENCE_THRESHOLD:
            logger.info(
                f"[Vision] page={page_num} chart_graph confidence {confidence:.2f} ≤ {GRAPH_CONFIDENCE_THRESHOLD} "
                "— downgrading to unknown, no graph chunk created."
            )
            return {
                "type":               "unknown",
                "content_type":       CONTENT_TYPE_UNKNOWN,
                "classification":     "unknown",
                "confidence":         0.0,
                "summary":            "",
                "extracted_values":   {},
                "needs_manual_review": True,
                "vision_confidence":  0.0
            }

        # Fix 7: Mixed pages → text+table processing only, no graph chunk
        if classification == "mixed":
            logger.info(f"[Vision] page={page_num} classified as mixed — text+table extraction only, no graph chunk.")
            return {
                "type":               "mixed",
                "content_type":       CONTENT_TYPE_MIXED,
                "classification":     "mixed",
                "confidence":         confidence,
                "summary":            "",
                "extracted_values":   {},
                "needs_manual_review": False,
                "vision_confidence":  confidence
            }

        # Fix 9: text_document and image classification — no VL Stage 2 needed
        if classification in ("text_document", "image", "unknown"):
            return {
                "type":               classification,
                "content_type":       content_type,
                "classification":     classification,
                "confidence":         confidence,
                "summary":            "",
                "extracted_values":   {},
                "needs_manual_review": classification == "unknown",
                "vision_confidence":  confidence
            }

        # ---- Stage 2 — Deep extraction for table / chart_graph / flowchart ----
        # Only reached when classification in {table, chart_graph, flowchart}
        # and confidence > GRAPH_CONFIDENCE_THRESHOLD (for chart_graph)
        stage2_result = await self._stage2_extract(base64_image, classification, page_num)

        logger.info(
            f"[Vision] page={page_num} final: classification={classification} "
            f"content_type={content_type} confidence={confidence:.2f}"
        )

        return {
            "type":               classification,
            "content_type":       content_type,
            "classification":     classification,
            "confidence":         confidence,
            "summary":            stage2_result.get("summary", ""),
            "extracted_values":   stage2_result.get("extracted_values", {}),
            "needs_manual_review": False,
            "vision_confidence":  confidence
        }

    # ------------------------------------------------------------------
    # Fix 4: Fallback — never fabricate content
    # ------------------------------------------------------------------
    def _fallback_analysis(self, page_type: str, page_num: int) -> Dict[str, Any]:
        """
        Safe fallback when the VL API is completely unavailable.
        Returns empty fields — NEVER invents chart summaries or trend lines.
        """
        logger.warning(
            f"[Vision Fallback] page={page_num} type={page_type} — "
            "VL unavailable. Returning empty result. No content fabricated."
        )
        return {
            "type":               "unknown",
            "content_type":       CONTENT_TYPE_UNKNOWN,
            "classification":     "unknown",
            "confidence":         0.0,
            "summary":            "",
            "extracted_values":   {},
            "needs_manual_review": True,
            "vision_confidence":  0.0
        }


vision_pipeline = VisionPipeline()
