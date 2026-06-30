import json
import logging
from typing import List, Dict, Any, TypedDict, Annotated, Optional, Union
import httpx
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph.graph import StateGraph, END
from backend.app.config import settings
from backend.app.models import AgentTrace, DocumentPage, Document
from backend.app.services.embeddings import embedding_service
from backend.app.services.vector_store import qdrant_store
from backend.app.services.search_store import search_store
from backend.app.database import AsyncSessionLocal
from sqlalchemy import select
from backend.app.services.reranker import reranker
import time
import hashlib

logger = logging.getLogger("app.agent_workflow")

# Maximum retries to prevent infinite loops
MAX_RETRIES = 1

ENUMERATION_VERBS = {
    "list",
    "enumerate",
    "identify",
    "provide",
    "show",
    "mention",
    "state",
    "describe",
    "summarize",
    "extract"
}

ENUMERATION_NOUNS = {
    "requirements",
    "provisions",
    "changes",
    "amendments",
    "relaxations",
    "exemptions",
    "conditions",
    "obligations",
    "timelines",
    "deadlines",
    "disclosures",
    "penalties",
    "fees",
    "limits",
    "thresholds",
    "rates",
    "procedures",
    "categories",
    "participants",
    "order types"
}

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    query: str
    session_id: str
    filters: Optional[Dict[str, Any]]
    structured_query: Dict[str, Any]
    search_tasks: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    vision_results: List[Dict[str, Any]]
    validation_result: Dict[str, Any]
    answer: Union[str, Dict[str, Any]]   # str for prose, dict for structured table
    citations: List[Dict[str, Any]]
    critic_result: Dict[str, Any]
    retry_count: int
    trace_steps: List[Dict[str, Any]]
    execution_times: Dict[str, float]
    reranker_scores: List[float]
    final_prompt: str

# Helper for calling Qwen3 Ollama API
async def call_qwen(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    json_format: bool = False,
    timeout_seconds: float = 300.0,
) -> str:
    """Helper method to invoke Ollama Qwen3 model."""
    url = f"{settings.llm.ollama_url}/api/chat"
    payload = {
        "model": settings.llm.reasoning_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "options": {
            "temperature": settings.llm.temperature
        },
        "stream": False
    }
    if json_format:
        payload["format"] = "json"

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                logger.warning(f"Ollama Qwen returned status {response.status_code}. Using fallback.")
    except Exception as e:
        logger.exception(f"Ollama call failed. Triggering fallback. Exception: {e}")
    
    # Fallback simulation
    return simulate_llm_response(prompt, json_format)

def simulate_llm_response(prompt: str, json_format: bool) -> str:
    """Simulates LLM response if service is offline."""
    p_lower = prompt.lower()
    if json_format:
        if "query" in p_lower or "understanding" in p_lower:
            kws = []
            if "margin" in p_lower:
                kws.append("margin")
            if "derivative" in p_lower or "currency" in p_lower:
                kws.append("derivatives")
            if "order" in p_lower:
                kws.append("order types")
            return json.dumps({
                "intent": "compliance_search",
                "ambiguity": "low",
                "missing_dates": False,
                "clean_query": "compliance search circular requirements",
                "keywords": kws or ["compliance"]
            })
        elif "retrieval strategy" in p_lower or "planning" in p_lower:
            return json.dumps({
                "tasks": ["margin collection requirements", "f&o compliance rules"]
            })
        elif "evidence" in p_lower or "validation" in p_lower:
            return json.dumps({
                "sufficient": True,
                "missing": "",
                "contradiction": False
            })
        elif "critic" in p_lower:
            return json.dumps({
                "pass": True,
                "reason": "All checks validated."
            })
        return "{}"
    else:
        if "synthesize" in p_lower or "summarize" in p_lower:
            return (
        "I could not generate a response because the LLM call failed. "
        "Please retry."
                    )
        return "Simulated response from Assistant."

async def persist_trace(session_id: str, query: str, step_name: str, input_state: Dict[str, Any], output_state: Dict[str, Any]):
    """Persists intermediate agent steps in PostgreSQL for execution tracing."""
    try:
        async with AsyncSessionLocal() as db:
            trace = AgentTrace(
                session_id=session_id,
                query=query,
                step_name=step_name,
                input_state=input_state,
                output_state=output_state
            )
            db.add(trace)
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to persist agent trace: {e}")

# --- Agent 1: Query Understanding Agent ---
async def query_understanding_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 1: Query Understanding")
    prompt = (
        f"Analyze this search query: '{state['query']}'\n"
        "Extract intent, determine if it is ambiguous, identify if it lacks specific dates, produce a clean search prompt, and extract key query terms/keywords.\n"
        "Format output as JSON: {\"intent\": \"...\", \"ambiguity\": \"high|medium|low\", \"missing_dates\": true|false, \"clean_query\": \"...\", \"keywords\": [\"word1\", \"word2\"]}"
    )
    res = await call_qwen(prompt, "You are an expert query classifier for financial regulations.", json_format=True)
    try:
        structured = json.loads(res)
    except:
        structured = {"intent": "compliance_search", "ambiguity": "low", "missing_dates": False, "clean_query": state["query"], "keywords": [state["query"]]}

    step_data = {"name": "Query Understanding", "output": structured}
    state["structured_query"] = structured
    state["trace_steps"].append(step_data)
    
    await persist_trace(state["session_id"], state["query"], "Query Understanding", {"query": state["query"]}, structured)
    return state

# --- Agent 2: Planning Agent ---
async def planning_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 2: Planning")
    clean_q = state["structured_query"].get("clean_query", state["query"])
    prompt = (
        f"Given this analyzed query: '{clean_q}'\n"
        "Generate 1 to 3 targeted retrieval sub-queries for vector and full-text search.\n"
        "For regulatory or compliance questions, prioritize precise, section-oriented sub-queries rather than broad semantic paraphrases.\n"
        "Examples of strong sub-queries: 'margin collection requirements currency derivatives', 'exemptions and relaxations margin collection', 'definition of order types'.\n"
        "Format output as JSON: {\"tasks\": [\"task 1\", \"task 2\"]}"
    )
    res = await call_qwen(prompt, "You are a retrieval planner. Generate targeted search queries.", json_format=True)
    try:
        tasks = json.loads(res).get("tasks", [clean_q])
    except:
        tasks = [clean_q]

    if not tasks:
        tasks = [clean_q]

    step_data = {"name": "Planning", "output": {"tasks": tasks}}
    state["search_tasks"] = tasks
    logger.info("[Planning] Generated search tasks: %s", tasks)
    state["trace_steps"].append(step_data)
    
    await persist_trace(state["session_id"], state["query"], "Planning", {"structured_query": state["structured_query"]}, {"tasks": tasks})
    return state

# --- Filter Key Translator (Bug 7 fix) ---
# The public API accepts e.g. {"content_type": "table"} but Qdrant/ES payloads
# store the same concept under the key "type".  This map normalises before search.
FILTER_KEY_MAP: Dict[str, str] = {
    "content_type": "type",
}

def translate_filters(raw_filters: Dict[str, Any]) -> Dict[str, Any]:
    """Translate user-facing filter keys to the internal payload field names and remove empty/unsupported values."""
    translated: Dict[str, Any] = {}
    for k, v in raw_filters.items():
        key = FILTER_KEY_MAP.get(k, k)
        if v is None:
            continue
        if isinstance(v, dict):
            if v:
                logger.warning(f"Dropping unsupported filter value for '{key}': nested dicts are not supported.")
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (list, tuple)) and len(v) == 0:
            continue
        translated[key] = v
    return translated

# --- Agent 3: Hybrid Retrieval Agent ---
async def hybrid_retrieval_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 3: Hybrid Retrieval")
    start_time = time.time()
    tasks = state["search_tasks"] or [state["query"]]
    # Bug 7 fix: translate API filter keys (e.g. content_type → type)
    filters = translate_filters(state["filters"] or {})
    logger.info(f"[Hybrid Retrieval] raw_filters={state['filters']} translated_filters={filters}")
    query_lower = state["query"].lower()
    query_words = set(query_lower.split())
    is_enum_query = (
        any(v in query_lower for v in ENUMERATION_VERBS)
        or any(n in query_lower for n in ENUMERATION_NOUNS)
    )

    retry = state.get("retry_count", 0)
    if is_enum_query:
        qdrant_limit = 100 + (retry * 20)
        es_limit = 50 + (retry * 10)
        rerank_limit = 30 + (retry * 10)
    else:
        qdrant_limit = 50 + (retry * 20)
        es_limit = 20 + (retry * 10)
        rerank_limit = 8 + (retry * 4)

    # ------------------------------------------------------------------
    # P0: Subject/Title query shortcut
    # When the user asks "what is the subject / title of this circular",
    # we inject a dedicated search for the subject line text so it always
    # surfaces, regardless of whether chunks have the new schema.
    # ------------------------------------------------------------------
    SUBJECT_QUERY_SIGNALS = {
        "subject", "title", "circular", "regarding", "sub", "re:", "topic"
    }
    query_words_set = set(state["query"].lower().split())
    is_subject_query = bool(query_words_set & SUBJECT_QUERY_SIGNALS)

    # Also check for explicit patterns like "what is the subject"
    _ql = state["query"].lower()
    if any(p in _ql for p in ["what is the subject", "what is the title", "subject of this", "title of this", "subject of the", "title of the"]):
        is_subject_query = True

    if is_subject_query:
        # Add an explicit sub-query that targets subject-line text directly
        subject_task = "Sub: Subject: circular subject heading title"
        if subject_task not in tasks:
            tasks = [subject_task] + list(tasks)
        logger.info(f"[Hybrid Retrieval] Subject-query shortcut active — injected subject search task.")

    merged_chunks = []
    seen_ids = set()

    for task in tasks:
        logger.info(f"[Hybrid Retrieval] Processing task: '{task[:80]}'")

        # Embed search task query
        vector = await embedding_service.get_embedding(task)

        # 1. Retrieve from Qdrant vector store
        qdrant_results = await qdrant_store.search_collection(vector, limit=qdrant_limit, filters=filters)
        logger.info(f"[Hybrid Retrieval] Qdrant returned {len(qdrant_results)} results for task")

        # 2. Retrieve from Elasticsearch full-text search
        es_results = await search_store.search_chunks(task, limit=es_limit, filters=filters)
        logger.info(f"[Hybrid Retrieval] Elasticsearch returned {len(es_results)} results for task")

        # Merge results (deduplicate by SHA-256 of text)
        all_results = qdrant_results + es_results
        for r in all_results:
            payload = r.get("payload", {})
            text = payload.get("text", "")
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()

            if chunk_hash not in seen_ids:
                seen_ids.add(chunk_hash)
                page_val = payload.get("page_number") or payload.get("page") or 1
                # Provide richer context to the reranker by including title and section
                reranker_text = f"TITLE: {payload.get('title') or ''}\nSECTION: {payload.get('section') or ''}\n{text}"
                merged_chunks.append({
                    "title": payload.get("title") or "Unknown Title",
                    "doc_id": payload.get("doc_id"),
                    "page": page_val,
                    "section": payload.get("section") or "Unknown Section",
                    "text": text or "",
                    "reranker_text": reranker_text,
                    "circular_number": payload.get("circular_number") or "",
                    "document_version": payload.get("document_version") or payload.get("version") or 1,
                    "upload_date": payload.get("upload_date"),
                    "score": r.get("score", 0.5)
                })

    logger.info(f"[Hybrid Retrieval] Total merged (dedup) chunks across all tasks: {len(merged_chunks)}")

    # Pre-rerank filter: Remove boilerplate/garbage chunks before scoring
    BOILERPLATE_PATTERNS = ["go to index", "continuation sheet", "p a g e", "part – a", "part - a"]
    MIN_CHUNK_WORDS = 5
    filtered_chunks = []
    removed_boilerplate = 0
    for c in merged_chunks:
        text = c.get("text", "").lower()
        word_count = len(text.split())
        is_boilerplate = any(pat in text for pat in BOILERPLATE_PATTERNS)
        if is_boilerplate or word_count < MIN_CHUNK_WORDS:
            removed_boilerplate += 1
            continue
        filtered_chunks.append(c)
    if removed_boilerplate:
        logger.info(
            f"[Hybrid Retrieval] Pre-rerank filter removed {removed_boilerplate} boilerplate/short chunks. "
            f"Remaining: {len(filtered_chunks)}"
        )
    merged_chunks = filtered_chunks if filtered_chunks else merged_chunks

    # P0/Fix 3: Metadata Weighted Retrieval — boost subject/title chunks for subject-type queries
    TITLE_QUERY_KEYWORDS = {"subject", "title", "circular", "issued", "date", "number", "regarding", "sub"}
    query_lower_words = set(state["query"].lower().split())
    is_title_query = bool(query_lower_words & TITLE_QUERY_KEYWORDS)
    if is_title_query:
        boosted_count = 0
        for c in merged_chunks:
            payload_section = (c.get("section") or "").lower()
            text_lower_c    = (c.get("text") or "").lower()
            is_subject_chunk = (
                payload_section in ("subject", "title")
                or "sub:" in text_lower_c[:60]
                or "subject:" in text_lower_c[:60]
            )
            if is_subject_chunk:
                c["score"] = c.get("score", 0.5) + 0.25
                boosted_count += 1
        if boosted_count:
            logger.info(f"[Hybrid Retrieval] Metadata boost applied to {boosted_count} subject/title chunks for title-type query.")

    logger.info(f"[Hybrid Retrieval] Chunks entering reranker: {len(merged_chunks)}")
    # Rerank with CrossEncoder
    merged_chunks = reranker.rerank(state["query"], merged_chunks, top_k=rerank_limit)
    logger.info(f"[Hybrid Retrieval] Chunks after rerank (top_k={rerank_limit}): {len(merged_chunks)}")

    # P0: After CrossEncoder reranking, pin subject/title chunks to top for subject queries.
    # CrossEncoder is trained on relevance, not doc structure — it may rank body text higher
    # than the subject line. This override fixes that for both old-schema and new-schema chunks.
    if is_title_query:
        pinned = 0
        for c in merged_chunks:
            text_lower_c    = (c.get("text") or "").lower()
            payload_section = (c.get("section") or "").lower()
            is_subject_chunk = (
                payload_section in ("subject", "title")
                or "sub:" in text_lower_c[:60]
                or "subject:" in text_lower_c[:60]
            )
            if is_subject_chunk:
                # Override rerank_score to guaranteed top position
                c["rerank_score"] = 10.0 + c.get("score", 0.5)
                c["importance_score"] = 1.0
                pinned += 1
        if pinned:
            # Re-sort after pinning so pinned chunks actually appear at the top
            merged_chunks = sorted(merged_chunks, key=lambda x: x.get("rerank_score", -999.0), reverse=True)
            logger.info(f"[Hybrid Retrieval] Pinned {pinned} subject/title chunk(s) to top via rerank_score override and re-sorted results.")


    # Detailed chunk-level logging
    reranker_scores = []
    logger.info("=== RETRIEVED CHUNKS DETAILED LOGGING ===")
    for idx, chunk in enumerate(merged_chunks, 1):
        similarity_score = chunk.get("score", "N/A")
        rerank_score = chunk.get("rerank_score", "N/A")
        reranker_scores.append(rerank_score)
        logger.info(f"[CHUNK {idx}]")
        logger.info(f"  PAGE       : {chunk.get('page', 'N/A')}")
        logger.info(f"  TITLE      : {chunk.get('title', 'Unknown Title')}")
        logger.info(f"  SECTION    : {chunk.get('section', 'Unknown Section')}")
        logger.info(f"  SCORE      : {similarity_score}")
        logger.info(f"  RERANK     : {rerank_score}")
        logger.info(f"  TEXT_PREVIEW: {chunk.get('text', '')[:120]}...")
    logger.info("========================================")

    state["reranker_scores"] = reranker_scores
    if "execution_times" not in state:
        state["execution_times"] = {}
    state["execution_times"]["hybrid_retrieval"] = time.time() - start_time

    step_data = {"name": "Hybrid Retrieval", "output": {"chunk_count": len(merged_chunks)}}
    logger.info("==== HYBRID DEBUG ====")
    logger.info(f"merged_chunks count = {len(merged_chunks)}")

    if merged_chunks:
        logger.info(f"chunk keys = {list(merged_chunks[0].keys())}")
        logger.info(f"text preview = {merged_chunks[0].get('text', '')[:200]}")
    
    state["retrieved_chunks"] = merged_chunks
    state["trace_steps"].append(step_data)

    await persist_trace(
        state["session_id"], state["query"], "Hybrid Retrieval",
        {"tasks": tasks, "filters": filters},
        {"chunk_count": len(merged_chunks)}
    )
    return state

import hashlib

# --- Agent 4: Validation Agent ---
async def validation_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 4: Validation")
    start_time = time.time()

    chunks = state["retrieved_chunks"]
    logger.info("==== VALIDATION DEBUG ====")
    logger.info(f"chunks count = {len(chunks)}")

    if chunks:
        logger.info(f"top chunk keys = {list(chunks[0].keys())}")
        logger.info(f"top rerank score = {chunks[0].get('rerank_score')}")
        logger.info(f"text preview = {chunks[0].get('text', '')[:200]}")
    
    # Handle empty retrieval
    if not chunks:
        logger.warning("No chunks retrieved.")

        val = {
            "sufficient": False,
            "missing": "No context retrieved",
            "contradiction": False,
            "relevant": False
        }

        # Increment retry count so MAX_RETRIES guard works even with 0 chunks
        state["retry_count"] = state.get("retry_count", 0) + 1
        logger.info(f"Incremented retry count (no chunks) to {state['retry_count']}")

        state["validation_result"] = val
        state["trace_steps"].append({
            "name": "Validation",
            "output": val
        })

        return state

    evidence_text = "\n".join(
        [
            f"Source {idx+1}: {c['text']}"
            for idx, c in enumerate(chunks)
        ]
    )

    # ----------------------------
    # Keyword Coverage Check
    # ----------------------------
    # Remove stop words to focus on important keywords
    # IMPORTANT: strip punctuation BEFORE checking stop words so "segment." -> "segment" -> filtered
    STOP_WORDS = {
        "explain", "available", "segment", "show", "list",
        "give", "what", "which", "describe", "all",
        "tell", "please", "details", "about", "can",
        "is", "are", "the", "a", "and", "or", "in",
        "on", "at", "to", "for", "of", "from", "by"
    }

    query_words = set()
    for w in state["query"].split():
        # Strip punctuation FIRST, then check stop words
        cleaned = w.lower().strip(".,?!;:'\"")
        if len(cleaned) > 3 and cleaned not in STOP_WORDS:
            query_words.add(cleaned)

    evidence_lower = evidence_text.lower()

    matched_words = [
        word for word in query_words
        if word in evidence_lower
    ]

    retrieved_chunks = chunks
    coverage = len(matched_words) / len(query_words) if query_words else 1.0

    # Bug 2 fix: use .get() with default 0.0 — prevents KeyError when reranker
    # failed to load and chunks were returned without a rerank_score key.
    top_rerank = retrieved_chunks[0].get("rerank_score", 0.0)

    validation_pass = (
        top_rerank > 0.75
        or coverage > 0.55
    )

    # Extended validation logging
    logger.info(f"[Validation] State keys visible: {list(state.keys())}")
    logger.info(f"[Validation] retrieved_chunks count visible to validator: {len(chunks)}")
    logger.info(f"[Validation] top_rerank_score: {top_rerank:.4f}")
    logger.info(f"[Validation] query_words: {query_words}")
    logger.info(f"[Validation] matched_words: {matched_words}")
    logger.info(f"[Validation] coverage: {coverage:.2f}")

    keyword_mismatch = coverage < 0.5

    # ----------------------------
    # Semantic Validation
    # ----------------------------
    prompt = f"""
User Query:
{state['query']}

Retrieved Context:
{evidence_text}

Determine:

1. Does the retrieved context answer the query?
2. Is important information missing?
3. Are the retrieved chunks unrelated?
4. Is there contradiction?

Return ONLY JSON:

{{
"sufficient": true,
"missing": "",
"contradiction": false,
"relevant": true
}}
"""

    try:
        res = await call_qwen(
            prompt,
            "You are a strict fact checking validator.",
            json_format=True
        )

        val = json.loads(res)

    except Exception as e:
        # Bug 3 fix: a crashed validation must NOT grant passage to Synthesis.
        # The old default (sufficient=True) caused hallucination on every LLM
        # timeout or JSON parse error.  Now we force a retry instead.
        logger.exception(
            f"[Validation] Validation LLM call failed — forcing sufficient=False "
            f"so the graph retries retrieval rather than synthesising blind. Error: {e}"
        )
        val = {
            "sufficient": False,
            "missing": "Validation LLM call failed; retrieval will be retried.",
            "contradiction": False,
            "relevant": False
        }

    # Force retry if keywords don't match (hard keyword mismatch)
    if keyword_mismatch:
        logger.warning(
            f"Keyword mismatch detected. Coverage={coverage:.2f}. Forcing retry."
        )
        val["sufficient"] = False
        val["relevant"] = False

    # Final safety: if not relevant, must not be sufficient
    if not val.get("relevant", True):
        val["sufficient"] = False

    # OVERRIDE: If coverage >= 0.5 AND we have chunks, trust coverage over LLM validation.
    # This prevents LLM from causing infinite loops when retrieval quality is actually acceptable.
    if not keyword_mismatch and chunks and not val.get("sufficient", True):
        logger.warning(
            f"LLM validation returned insufficient=True but coverage={coverage:.2f} >= 0.5 with "
            f"{len(chunks)} chunks. Overriding LLM validation to allow synthesis."
        )
        val["sufficient"] = True
        val["relevant"] = True

    # Trace
    step_data = {
        "name": "Validation",
        "output": val,
        "keyword_coverage": coverage,
        "matched_words": matched_words
    }

    # Increment retry_count inside the node if validation fails
    should_retry = not val.get("sufficient", True)
    if should_retry:
        state["retry_count"] = state.get("retry_count", 0) + 1
        logger.info(f"Incremented retry count in Validation Agent to {state['retry_count']}")

    state["validation_result"] = val
    state["trace_steps"].append(step_data)

    if "execution_times" not in state:
        state["execution_times"] = {}

    state["execution_times"]["validation"] = (
        time.time() - start_time
    )

    await persist_trace(
        state["session_id"],
        state["query"],
        "Validation",
        {
            "chunks_count": len(chunks),
            "coverage": coverage,
            "matched_words": matched_words
        },
        val
    )

    return state

# --- Agent 5: Vision Agent ---
async def vision_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 5: Vision")
    # This agent runs if graphs/tables/scans are requested or present in retrieved context
    vision_records = []
    
    # Check if user query mentions figures, tables, graphs or if retrieved chunks include
    # structured/visual content types that merit vision-aware reasoning.
    # Bug 8 fix: broaden triggers to include inferred table/graph/mixed content.
    query_lower = state["query"].lower()
    need_vision = any(kw in query_lower for kw in [
        "graph", "chart", "table", "figure", "revenue", "trend",
        "trading", "commodity", "schedule", "annexure", "rate", "fee",
        "timing", "hours", "percentage", "limit", "threshold", "extract",
        "image", "visual", "diagram", "flowchart"
    ])
    need_vision = need_vision or any(
        (c.get("content_type") or c.get("type") or "").lower() in {
            "table", "graph", "mixed", "image", "flowchart"
        }
        for c in state["retrieved_chunks"]
    )
    
    if need_vision:
        # Query database to find any document pages containing table/graph layouts for the retrieved documents
        doc_ids = list(set([c["doc_id"] for c in state["retrieved_chunks"] if c.get("doc_id")]))
        if doc_ids:
            try:
                async with AsyncSessionLocal() as db:
                    stmt = select(DocumentPage).where(
                        DocumentPage.document_id.in_(doc_ids),
                        DocumentPage.classification.in_(["table", "graph", "scanned", "mixed"])
                    )
                    res = await db.execute(stmt)
                    pages = res.scalars().all()
                    
                    for p in pages:
                        vision_records.append({
                            "doc_id": str(p.document_id),
                            "page": p.page_number,
                            "classification": p.classification,
                            "summary": p.vision_summary,
                            "values": p.vision_extracted_values
                        })
            except Exception as e:
                logger.error(f"Vision Agent DB fetch failed: {e}")

    # Bug 8 logging: always surface what the Vision Agent actually found
    doc_ids_for_log = list(set([c["doc_id"] for c in state["retrieved_chunks"] if c.get("doc_id")]))
    logger.info(
        f"[Vision Agent] need_vision={need_vision}, doc_ids_checked={doc_ids_for_log}, "
        f"db_vision_records_found={len(vision_records)}"
    )
    if need_vision and doc_ids_for_log and not vision_records:
        logger.warning(
            "[Vision Agent] Vision triggered by query keywords but no pre-stored vision summaries "
            "found in DB for these documents. Pages were likely classified as 'text' during ingestion. "
            "Consider re-ingesting the document or calling POST /document/{id}/reindex-metadata."
        )

    step_data = {"name": "Vision Agent", "output": {"records_fetched": len(vision_records)}}
    state["vision_results"] = vision_records
    state["trace_steps"].append(step_data)

    await persist_trace(
        state["session_id"], state["query"], "Vision Agent",
        {"need_vision": need_vision, "doc_ids": doc_ids_for_log},
        {"records_fetched": len(vision_records)}
    )
    return state

async def find_related_documents(db: AsyncSession, query_keywords: List[str]) -> List[Dict[str, Any]]:
    """
    Searches the database for other historical documents containing overlapping
    keywords, topics, entities, or financial terms.
    """
    if not query_keywords:
        return []
        
    try:
        stmt = select(Document).where(Document.status == "active")
        res = await db.execute(stmt)
        docs = res.scalars().all()
        
        related = []
        query_kw_set = set(k.lower() for k in query_keywords)
        for doc in docs:
            doc_kws = (doc.keywords or []) + (doc.topics or []) + (doc.entities or []) + (doc.financial_terms or [])
            doc_kws_set = set(k.lower() for k in doc_kws)
            overlap = query_kw_set.intersection(doc_kws_set)
            if overlap:
                related.append({
                    "id": str(doc.id),
                    "name": doc.name,
                    "circular_number": doc.circular_number or "N/A",
                    "keywords": doc.keywords or [],
                    "topics": doc.topics or [],
                    "entities": doc.entities or [],
                    "financial_terms": doc.financial_terms or [],
                    "circular_type": doc.circular_type or "N/A"
                })
        return related
    except Exception as e:
        logger.error(f"Error finding related documents: {e}")
        return []

# --- Agent 6: Synthesis Agent ---
async def synthesis_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 6: Synthesis")
    start_time = time.time()
    chunks = state["retrieved_chunks"]
    vision = state["vision_results"]
    
    # 2. Sort chunks by descending rerank score before generation
    # If the numeric trigger applies, we will prioritize chunks containing digits.
    query_lower = state["query"].lower()
    has_numeric_trigger = any(
        kw in query_lower 
        for kw in ["error code", "code", "quantity", "percentage", "date", "time", "limit", "threshold"]
    )
    
    def get_sort_key(c):
        score = c.get("rerank_score", c.get("score", 0.0))
        if has_numeric_trigger:
            has_number = any(char.isdigit() for char in c.get("text", ""))
            return (1 if has_number else 0, score)
        return (0, score)
    
    sorted_chunks = sorted(chunks, key=get_sort_key, reverse=True)

    # Fix 11: Adaptive top-K based on query intent
    def _compute_adaptive_k(query: str) -> int:
        q = query.lower()
        query_words = set(q.split())
        is_enum_query = (
            any(v in q for v in ENUMERATION_VERBS)
            or any(n in q for n in ENUMERATION_NOUNS)
        )
        if is_enum_query:
            return 15
        if any(kw in q for kw in ["compare", "difference", "analyse", "analysis", "reasoning", "explain all", "detailed"]):
            return 10
        if any(kw in q for kw in ["summarize", "summary", "overview", "list all", "enumerate"]):
            return 10
        if any(kw in q for kw in ["table", "annexure", "schedule", "timing", "trading hours"]):
            return 7
        # Default short factoid
        return 8

    adaptive_k = _compute_adaptive_k(state["query"])
    logger.info(f"[Synthesis] Adaptive top-K selected: K={adaptive_k} for query type.")

    # Fix 13: Protect subject/title chunks — mark them before quality filter
    for c in sorted_chunks:
        text_low = (c.get("text") or "").lower()
        section_low = (c.get("section") or "").lower()
        if (
            c.get("importance_score", 0.0) >= 1.0
            or section_low in ("subject", "title")
            or "sub:" in text_low[:50]
            or "subject:" in text_low[:50]
        ):
            c["_protected"] = True
        else:
            c["_protected"] = False

    # Fix 12: Remove low-quality chunks (but never remove protected subject/title chunks)
    RERANK_FLOOR = -8.0
    quality_filtered = []
    for c in sorted_chunks:
        text_raw = c.get("text", "")
        vis_conf = float(c.get("vision_confidence", 1.0))  # default 1.0 for non-visual
        ctype    = (c.get("content_type") or c.get("type") or "").upper()

        if c.get("_protected"):
            quality_filtered.append(c)
            continue
        if len(text_raw) < 50:
            logger.debug(f"[Synthesis] Discarding short chunk (len={len(text_raw)})")
            continue
        if "fallback extraction" in text_raw.lower():
            logger.warning("[Synthesis] Discarding fabricated fallback chunk.")
            continue
        if ctype == "GRAPH" and vis_conf < 0.8:
            logger.info(f"[Synthesis] Discarding low-confidence GRAPH chunk (conf={vis_conf:.2f}).")
            continue
        if c.get("rerank_score", 0.0) <= RERANK_FLOOR:
            continue
        quality_filtered.append(c)

    if not quality_filtered and sorted_chunks:
        quality_filtered = sorted_chunks[:1]   # last-resort fallback

    top_chunks = quality_filtered[:adaptive_k]

    logger.info(
        f"[Synthesis] chunks_in={len(chunks)}, sorted={len(sorted_chunks)}, "
        f"after_quality_filter={len(quality_filtered)}, "
        f"top_chunks_K{adaptive_k}={len(top_chunks)}, "
        f"rerank_scores={[round(c.get('rerank_score', 0.0), 3) for c in top_chunks]}"
    )

    # Update state so Citation Agent works on the same chunk set
    state["retrieved_chunks"] = top_chunks

    context_blocks = []
    for c in top_chunks:
        text = c["text"]
        if has_numeric_trigger and any(char.isdigit() for char in text):
            text = f"[NOTE: The context contains an explicit value. Use it directly if relevant.]\n{text}"
            c["text"] = text
        section_label = c.get("section") or "Unknown Section"
        title = c.get("title", "Unknown Title")
        page_no = c.get("page", "?")
        context_blocks.append(
            f"### SECTION {section_label} | Page {page_no} | Document {title}\n{text}"
        )

    for v in vision:
        context_blocks.append(
            f"Vision data on Page {v['page']} (Type: {v['classification']}): "
            f"Summary: {v['summary']}. Data: {json.dumps(v['values'])}"
        )

    # Find other related documents in the system history
    related_docs_str = ""
    try:
        async with AsyncSessionLocal() as db:
            query_keywords = state.get("structured_query", {}).get("keywords", [])
            related_docs = await find_related_documents(db, query_keywords)
            retrieved_doc_ids = set(c.get("doc_id") for c in top_chunks if c.get("doc_id"))
            filtered_related = [d for d in related_docs if d["id"] not in retrieved_doc_ids]
            if filtered_related:
                related_docs_str = "\n".join([
                    f"- Document: {d['name']} (Circular Number: {d['circular_number']}, "
                    f"Type: {d['circular_type']}) | Keywords: {', '.join(d['keywords'])}"
                    for d in filtered_related
                ])
    except Exception as e:
        logger.error(f"Failed to fetch related documents during synthesis: {e}")

    context = "\n\n".join(context_blocks)

    logger.info(f"[Synthesis] context_length_chars={len(context)}")
    logger.info(f"[Synthesis] First 1000 chars of context sent to qwen3:\n{context[:1000]}")

    system_prompt = (
        "You are an NSE compliance assistant.\n"
        "Answer ONLY using the provided context.\n"
        "\n"
        "If a number, error code, percentage, quantity, date, or identifier exists in the context, "
        "always include it explicitly.\n"
        "\n"
        "Never say information is unavailable if it appears in the context.\n"
        "\n"
        "Prefer exact values over summaries.\n"
        "\n"
        "Do not infer from similar sections.\n"
        "\n"
        "Do not combine unrelated chunks.\n"
        "\n"
        "Always cite sources with document title, page, and section.\n"
        "\n"
        "If the answer cannot be found, say:\n"
        "'Information not found in the provided documents.'"
    )
    # Check if the query is an extraction/enumeration query requesting lists/relaxations/exemptions/etc.
    query_lower_synth = state["query"].lower()
    is_extraction_query = (
        any(v in query_lower_synth for v in ENUMERATION_VERBS)
        or any(n in query_lower_synth for n in ENUMERATION_NOUNS)
    )
    if is_extraction_query:
        system_prompt += (
            "\n\nCRITICAL ENUMERATION RULE:\n"
            "This query is a request for listing/extracting items (e.g. relaxations, exemptions, changes, lists). "
            "You MUST scan all retrieved chunks carefully, identify EVERY applicable item, relaxation, exemption, or change "
            "present anywhere in the context, and extract it completely. "
            "Do NOT summarize them into high-level groups, do NOT skip details, and do NOT truncate the output. "
            "List every single distinct item and provide its citation."
        )

    # --- Table-aware structured synthesis ---
    # When the query targets tabular data (via filter or keywords) and we have
    # retrieved chunks, ask qwen3 to return a structured JSON table instead of prose.
    is_table_query = (
        (state.get("filters") or {}).get("content_type") == "table"
        or any(kw in query_lower_synth for kw in [
            "table", "annexure", "schedule", "trading hours", "timing",
            "commodity", "extract the", "list the"
        ])
    )

    if is_table_query and top_chunks:
        logger.info("[Synthesis] Table query detected — requesting structured JSON table from qwen3.")
        table_system_prompt = (
            "You are an NSE compliance assistant that extracts structured tabular data.\n"
            "Return ONLY a valid JSON object — no prose, no markdown fences.\n"
            "The JSON must have two keys:\n"
            "  \"table_title\": string — a short descriptive title for the table\n"
            "  \"rows\": array of objects — each object has column-header strings as keys "
            "and cell values as string values.\n"
            "If the context does not contain a table, still return the best structured "
            "representation of the data present."
        )
        if is_extraction_query:
            table_system_prompt += (
                "\n\nCRITICAL ENUMERATION RULE:\n"
                "Identify and extract EVERY applicable item from ALL retrieved chunks. "
                "Do not omit any items or rows present in the context."
            )
        table_prompt = (
            f"User request: {state['query']}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Extract ALL table rows from the context above and return as JSON."
        )
        table_raw = await call_qwen(table_prompt, table_system_prompt, json_format=True, timeout_seconds=600.0)
        try:
            table_data = json.loads(table_raw)
            # Validate it has the expected shape
            if isinstance(table_data, dict) and "rows" in table_data:
                logger.info(
                    f"[Synthesis] Structured table extracted: "
                    f"title='{table_data.get('table_title')}', "
                    f"rows={len(table_data.get('rows', []))}"
                )
                step_data = {"name": "Synthesis", "output": {"table_rows": len(table_data.get('rows', []))}}
                state["answer"] = table_data   # type: ignore[assignment]
                state["trace_steps"].append(step_data)
                if "execution_times" not in state:
                    state["execution_times"] = {}
                state["execution_times"]["synthesis"] = time.time() - start_time
                await persist_trace(
                    state["session_id"], state["query"], "Synthesis",
                    {"context_length": len(context), "mode": "table"},
                    {"table_rows": len(table_data.get('rows', []))}
                )
                return state
        except Exception as table_err:
            logger.warning(
                f"[Synthesis] Table JSON extraction failed ({table_err}). "
                "Falling back to prose synthesis."
            )
    # --- End table-aware block ---

    prompt = f"Question:\n{state['query']}\n\n"
    if related_docs_str:
        prompt += f"System History (Other Related Documents in Database):\n{related_docs_str}\n\n"
    prompt += (
        f"Retrieved Context:\n{context}\n\n"
        "Provide:\n"
        "1. Detailed answer\n"
        "2. Bullet points\n"
        "3. Source citations (Document Title, Page, Section)"
    )

    state["final_prompt"] = prompt
    res = await call_qwen(prompt, system_prompt, timeout_seconds=600.0)

    # Prevent hallucination: if qwen3 says it has no info, surface that cleanly
    if "information not found" in res.lower() or "could not find sufficient" in res.lower() or not res.strip():
        res = "Information not found in the provided documents."

    step_data = {"name": "Synthesis", "output": {"answer_length": len(res)}}
    state["answer"] = res
    state["trace_steps"].append(step_data)
    if "execution_times" not in state:
        state["execution_times"] = {}
    state["execution_times"]["synthesis"] = time.time() - start_time

    await persist_trace(
        state["session_id"], state["query"], "Synthesis",
        {"context_length": len(context), "mode": "prose"},
        {"answer_length": len(res)}
    )
    return state

# --- Agent 7: Citation Agent ---
async def citation_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 7: Citation")
    start_time = time.time()
    answer = state["answer"]
    logger.info(f"[Citation] Answer type: {type(answer).__name__}")
    chunks = state["retrieved_chunks"]

    # Bug 9 fix: if Synthesis ran with empty context (all chunks were filtered out),
    # retrieved_chunks will be empty here.  Any answer present is hallucinated —
    # override it with the safe no-answer string and return empty citations.
    if not chunks:
        logger.error(
            "[Citation] retrieved_chunks is EMPTY when Citation Agent runs. "
            "This means Synthesis executed without any real context. "
            "The answer is unreliable — replacing with safe no-answer response."
        )
        state["answer"] = "No relevant information found in the uploaded documents."
        state["citations"] = []
        state["trace_steps"].append({"name": "Citation", "output": {"citation_count": 0, "overridden": True}})
        if "execution_times" not in state:
            state["execution_times"] = {}
        state["execution_times"]["citation"] = time.time() - start_time
        await persist_trace(
            state["session_id"], state["query"], "Citation",
            {"answer": answer},
            {"citation_count": 0, "overridden": True}
        )
        return state

    citations = []
    vague_doc_names = []

    for c in chunks:
        title = c.get("title") or "Unknown Title"

        # Track vague titles
        if title == "Unknown Title":
            vague_doc_names.append(True)
            logger.warning(
                "[Citation] Retrieved chunk has vague document title. "
                "Ensure uploaded documents have a clear title on the first page."
            )

        citation = {
            "doc_id": c.get("doc_id"),
            "title": title,
            "page_number": c.get("page") or 1,
            "section": c.get("section") or "Unknown Section",
            "circular_number": c.get("circular_number") or "",
            "version": c.get("document_version") or 1
        }
        if citation not in citations:
            citations.append(citation)

    # If ALL chunks have vague titles, flag the validation result
    if vague_doc_names and len(vague_doc_names) == len(chunks):
        logger.error("[Citation] All retrieved chunks lack clear document titles. Cannot provide accurate citations.")
        state["validation_result"]["sufficient"] = False
        state["validation_result"]["missing"] = (
            "Document sources are not clearly identified. "
            "Please ensure documents contain clear titles on the first page."
        )

    logger.info(f"[Citation] Generated {len(citations)} citation(s) from {len(chunks)} chunk(s).")

    step_data = {"name": "Citation", "output": {"citation_count": len(citations)}}
    state["citations"] = citations
    state["trace_steps"].append(step_data)
    if "execution_times" not in state:
        state["execution_times"] = {}
    state["execution_times"]["citation"] = time.time() - start_time

    await persist_trace(
        state["session_id"], state["query"], "Citation",
        {"answer": answer},
        {"citation_count": len(citations)}
    )
    return state

# --- Agent 8: Critic Agent (Fix 14: smarter critic) ---
async def critic_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 8: Critic")

    answer    = state.get("answer", "")
    citations = state.get("citations", [])
    chunks    = state.get("retrieved_chunks", [])

    # Defensive: normalise answer to a string for all text-based checks.
    # Synthesis may return a dict for table queries; we serialise it here
    # so every downstream .lower() / signal-check never crashes.
    if isinstance(answer, dict):
        answer_text = json.dumps(answer, ensure_ascii=False)
        logger.info(f"[Critic] Answer type: dict — serialised for text checks (keys={list(answer.keys())})")
    else:
        answer_text = answer if isinstance(answer, str) else str(answer)
        logger.info(f"[Critic] Answer type: {type(answer).__name__}")

    # Fix 14: If we have a real answer, real citations, and real context, pass immediately.
    # Only escalate to LLM critic when there is material risk (no citations, empty answer,
    # or the answer itself contains typical hallucination signals).
    HALLUCINATION_SIGNALS = [
        "i cannot find", "i could not find", "no information", "not available in the documents",
        "invented", "fabricated", "not grounded", "no source", "i made up"
    ]
    answer_lower = answer_text.lower()

    fast_pass = (
        bool(answer)
        and bool(citations)
        and len(chunks) > 0
        and "information not found" not in answer_lower
        and not any(sig in answer_lower for sig in HALLUCINATION_SIGNALS)
    )

    if fast_pass:
        logger.info(
            "[Critic] Fast-pass: answer present, citations present, context non-empty, "
            "no hallucination signals detected — skipping LLM critic call."
        )
        critic_res = {"pass": True, "reason": "Fast-pass: clean answer with citations.", "loop_back_retrieval": False}
    else:
        prompt = (
            f"Evaluate this proposed compliance answer for issues:\n"
            f"Query: {state['query']}\n"
            f"Answer: {answer_text}\n"
            f"Citations: {json.dumps(citations)}\n\n"
            "Critique for 1) Hallucination risks, 2) Contradictions, 3) Outdated document versions. "
            "Set loop_back_retrieval=true ONLY when answer contains clearly invented facts not grounded in any source. "
            "Do NOT set loop_back_retrieval=true merely because the answer is short or cites few sources. "
            'Return output as JSON: {"pass": true|false, "reason": "...", "loop_back_retrieval": true|false}'
        )
        res = await call_qwen(prompt, "You are a critical quality control compliance auditor.", json_format=True)
        try:
            critic_res = json.loads(res)
        except:
            critic_res = {"pass": True, "reason": "Passed critic check automatically.", "loop_back_retrieval": False}

        # Fix 14: Only loop back when hallucination is explicitly flagged in the reason text
        reason_lower = critic_res.get("reason", "").lower()
        actual_hallucination = any(
            sig in reason_lower for sig in ["invented", "fabricated", "not grounded", "no source", "made up"]
        )
        if critic_res.get("loop_back_retrieval") and not actual_hallucination:
            logger.info(
                "[Critic] LLM requested loop_back_retrieval but no hallucination signal found in reason. "
                "Suppressing loop-back to avoid unnecessary re-retrieval."
            )
            critic_res["loop_back_retrieval"] = False

    # Increment retry_count only if actually looping back
    should_retry = not critic_res.get("pass", True) or critic_res.get("loop_back_retrieval", False)
    if should_retry:
        state["retry_count"] = state.get("retry_count", 0) + 1
        logger.info(f"Incremented retry count in Critic Agent to {state['retry_count']}")

    step_data = {"name": "Critic", "output": critic_res}
    state["critic_result"] = critic_res
    state["trace_steps"].append(step_data)

    await persist_trace(state["session_id"], state["query"], "Critic", {"answer": answer_text}, critic_res)
    return state

# --- Agent: No-Answer Terminal Node (Bug 4 fix) ---
async def no_answer_agent(state: AgentState) -> AgentState:
    """
    Terminal node reached when max retries are exhausted and retrieved_chunks
    is still empty.  Sets a safe answer and prevents Synthesis from ever
    running with empty context.
    """
    logger.warning(
        f"[NoAnswer] Max retries ({MAX_RETRIES}) exhausted with 0 usable chunks. "
        "Returning safe no-answer response — qwen3 will NOT be called."
    )
    state["answer"] = "No relevant information found in the uploaded documents."
    state["citations"] = []
    state["critic_result"] = {
        "pass": True,
        "reason": "No context available after all retries; safe no-answer returned.",
        "loop_back_retrieval": False
    }
    state["trace_steps"].append({
        "name": "NoAnswer",
        "output": {"reason": "empty_retrieval_after_max_retries"}
    })
    return state

# --- Routing Logic for LangGraph ---
def check_validation_route(state: AgentState) -> str:
    retry = state.get("retry_count", 0)
    validation = state.get("validation_result", {})
    chunks = state.get("retrieved_chunks", [])

    should_retry = not validation.get("sufficient", True)

    if should_retry and retry <= MAX_RETRIES:
        logger.info(
            f"[Validation→Router] Insufficient context. "
            f"Looping back to Hybrid Retrieval (Retry {retry}/{MAX_RETRIES})"
        )
        return "hybrid_retrieval"

    if should_retry:
        # Bug 4 fix: max retries exhausted — route to no_answer instead of
        # proceeding to vision_agent → synthesis with empty chunks.
        logger.warning(
            f"[Validation→Router] Max retries ({MAX_RETRIES}) reached with "
            f"{len(chunks)} chunk(s). Routing to no_answer terminal node."
        )
        return "no_answer"

    logger.info(
        f"[Validation→Router] Context sufficient ({len(chunks)} chunks). "
        "Proceeding to vision_agent."
    )
    return "vision_agent"

def check_critic_route(state: AgentState) -> str:
    critic = state.get("critic_result", {})
    retry = state.get("retry_count", 0)

    should_retry = not critic.get("pass", True) or critic.get("loop_back_retrieval", False)

    if should_retry and retry <= MAX_RETRIES:
        logger.info(
            f"[Critic→Router] Critic failed. Re-triggering retrieval (Retry {retry}/{MAX_RETRIES})"
        )
        return "hybrid_retrieval"

    if should_retry:
        logger.warning(
            f"[Critic→Router] Max retries ({MAX_RETRIES}) reached. "
            "Returning current answer despite critic issues."
        )

    return "end"

# Build Graph
builder = StateGraph(AgentState)

builder.add_node("query_understanding", query_understanding_agent)
builder.add_node("planning", planning_agent)
builder.add_node("hybrid_retrieval", hybrid_retrieval_agent)
builder.add_node("validation", validation_agent)
builder.add_node("vision_agent", vision_agent)
builder.add_node("synthesis", synthesis_agent)
builder.add_node("citation", citation_agent)
builder.add_node("critic", critic_agent)
# Bug 4 fix: terminal node that cleanly ends the graph when retrieval fails
builder.add_node("no_answer", no_answer_agent)

builder.set_entry_point("query_understanding")
builder.add_edge("query_understanding", "planning")
builder.add_edge("planning", "hybrid_retrieval")
builder.add_edge("hybrid_retrieval", "validation")

builder.add_conditional_edges(
    "validation",
    check_validation_route,
    {
        "hybrid_retrieval": "hybrid_retrieval",
        "vision_agent": "vision_agent",
        "no_answer": "no_answer",   # Bug 4 fix: safe terminal branch
    }
)

builder.add_edge("no_answer", END)   # no_answer terminates cleanly
builder.add_edge("vision_agent", "synthesis")
builder.add_edge("synthesis", "citation")
builder.add_edge("citation", "critic")

builder.add_conditional_edges(
    "critic",
    check_critic_route,
    {
        "hybrid_retrieval": "hybrid_retrieval",
        "end": END
    }
)

agent_workflow = builder.compile()

async def run_agent_search(query: str, session_id: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs the LangGraph orchestration flow and returns query search results."""
    initial_state = AgentState(
        query=query,
        session_id=session_id,
        filters=filters,
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[],
        vision_results=[],
        validation_result={},
        answer="",
        citations=[],
        critic_result={},
        retry_count=0,
        trace_steps=[],
        execution_times={},
        reranker_scores=[],
        final_prompt=""
    )
    
    final_state = await agent_workflow.ainvoke(initial_state)
    
    # Check if any citations have vague document titles
    warning = None
    vague_count = sum(1 for c in final_state["citations"] if c.get("title") == "Unknown Title")
    if vague_count > 0:
        warning = f"{vague_count} retrieved documents have unclear titles. Titles are generated from first-page content, so ensure uploaded documents have clear first-page headings."
    
    return {
        "query": final_state["query"],
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "agent_trace_session_id": final_state["session_id"],
        "warning": warning
    }
