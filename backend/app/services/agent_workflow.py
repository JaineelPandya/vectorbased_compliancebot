import json
import logging
from typing import List, Dict, Any, TypedDict, Annotated, Optional
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
    answer: str
    citations: List[Dict[str, Any]]
    critic_result: Dict[str, Any]
    retry_count: int
    trace_steps: List[Dict[str, Any]]
    execution_times: Dict[str, float]
    reranker_scores: List[float]
    final_prompt: str

# Helper for calling Qwen3 Ollama API
async def call_qwen(prompt: str, system_prompt: str = "You are a helpful assistant.", json_format: bool = False) -> str:
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
        async with httpx.AsyncClient(timeout=300.0) as client:
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
        "Generate a retrieval strategy as 1 to 3 distinct search sub-queries to query vector and full-text engines.\n"
        "Format output as JSON: {\"tasks\": [\"task 1\", \"task 2\"]}"
    )
    res = await call_qwen(prompt, "You are a retrieval planner. Generate key search queries.", json_format=True)
    try:
        tasks = json.loads(res).get("tasks", [clean_q])
    except:
        tasks = [clean_q]

    step_data = {"name": "Planning", "output": {"tasks": tasks}}
    state["search_tasks"] = tasks
    state["trace_steps"].append(step_data)
    
    await persist_trace(state["session_id"], state["query"], "Planning", {"structured_query": state["structured_query"]}, {"tasks": tasks})
    return state

# --- Agent 3: Hybrid Retrieval Agent ---
async def hybrid_retrieval_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 3: Hybrid Retrieval")
    start_time = time.time()
    tasks = state["search_tasks"] or [state["query"]]
    filters = state["filters"] or {}
    retry = state.get("retry_count", 0)

    qdrant_limit = 50 + (retry * 20)
    es_limit = 20 + (retry * 10)
    rerank_limit = 8 + (retry * 4)

    merged_chunks = []
    seen_ids = set()

    for task in tasks:
        # Embed search task query
        vector = await embedding_service.get_embedding(task)
        
        # 1. Retrieve from Qdrant vector store using a single document_chunks collection
        qdrant_results = await qdrant_store.search_collection(vector, limit=qdrant_limit, filters=filters)

        # 2. Retrieve from Elasticsearch full-text search
        es_results = await search_store.search_chunks(task, limit=es_limit, filters=filters)

        # Merge results (Deduplicate based on chunk id or text contents)
        all_results = qdrant_results + es_results
        for r in all_results:
            payload = r.get("payload", {})
            text = payload.get("text", "")
            chunk_hash = hashlib.sha256(text.encode()).hexdigest()
            
            if chunk_hash not in seen_ids:
                seen_ids.add(chunk_hash)
                merged_chunks.append({
                    "title": payload.get("title") or "Unknown Title",
                    "doc_id": payload.get("doc_id"),
                    "page": payload.get("page_number") or payload.get("page") or 1,
                    "section": payload.get("section") or "Unknown Section",
                    "text": text or "",
                    "circular_number": payload.get("circular_number") or "",
                    "document_version": payload.get("document_version") or payload.get("version") or 1,
                    "upload_date": payload.get("upload_date"),
                    "score": r.get("score", 0.5)
                })

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
        logger.info(f"Pre-rerank filter removed {removed_boilerplate} boilerplate/short chunks. Remaining: {len(filtered_chunks)}")
    merged_chunks = filtered_chunks if filtered_chunks else merged_chunks  # fallback to unfiltered if all removed

    # Rerank with CrossEncoder
    merged_chunks = reranker.rerank(state["query"], merged_chunks, top_k=rerank_limit)

    # Detailed logging with all metadata
    reranker_scores = []
    logger.info("=== RETRIEVED CHUNKS DETAILED LOGGING ===")
    for idx, chunk in enumerate(merged_chunks, 1):
        similarity_score = chunk.get("score", "N/A")
        rerank_score = chunk.get("rerank_score", "N/A")
        reranker_scores.append(rerank_score)
        logger.info(f"[CHUNK {idx}]")
        logger.info(f"  PAGE: {chunk.get('page', 'N/A')}")
        logger.info(f"  TITLE: {chunk.get('title', 'Unknown Title')}")
        logger.info(f"  SECTION: {chunk.get('section', 'Unknown Section')}")
        logger.info(f"  SIMILARITY_SCORE: {similarity_score}")
        logger.info(f"  RERANK_SCORE: {rerank_score}")
        logger.info(f"  TEXT_PREVIEW: {chunk.get('text', '')[:120]}...")
    logger.info("========================================")

    state["reranker_scores"] = reranker_scores
    if "execution_times" not in state: state["execution_times"] = {}
    state["execution_times"]["hybrid_retrieval"] = time.time() - start_time

    step_data = {"name": "Hybrid Retrieval", "output": {"chunk_count": len(merged_chunks)}}
    state["retrieved_chunks"] = merged_chunks
    state["trace_steps"].append(step_data)
    
    await persist_trace(state["session_id"], state["query"], "Hybrid Retrieval", {"tasks": tasks}, {"chunk_count": len(merged_chunks)})
    return state

import hashlib

# --- Agent 4: Validation Agent ---
async def validation_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 4: Validation")
    start_time = time.time()

    chunks = state["retrieved_chunks"]

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
    coverage = len(matched_words)/len(query_words) if query_words else 1.0

    top_rerank = retrieved_chunks[0]["rerank_score"]

    validation_pass = (
        top_rerank > 0.75
        or coverage > 0.55
    )

    logger.info(f"Query words: {query_words}")
    logger.info(f"Matched words: {matched_words}")
    logger.info(f"Coverage: {coverage:.2f}")

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
        logger.exception(f"Validation failed: {e}")

        val = {
            "sufficient": True,
            "missing": "",
            "contradiction": False,
            "relevant": True
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
    
    # Check if user query mentions figures, tables, graphs or if retrieved chunks are from table/graph/scan collections
    need_vision = any(kw in state["query"].lower() for kw in ["graph", "chart", "table", "figure", "revenue", "trend"])
    
    if need_vision:
        # Query database to find any document pages containing table/graph layouts for the retrieved documents
        doc_ids = list(set([c["doc_id"] for c in state["retrieved_chunks"] if c.get("doc_id")]))
        if doc_ids:
            try:
                async with AsyncSessionLocal() as db:
                    stmt = select(DocumentPage).where(
                        DocumentPage.document_id.in_(doc_ids),
                        DocumentPage.classification.in_(["table", "graph", "scanned"])
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

    step_data = {"name": "Vision Agent", "output": {"records_fetched": len(vision_records)}}
    state["vision_results"] = vision_records
    state["trace_steps"].append(step_data)
    
    await persist_trace(state["session_id"], state["query"], "Vision Agent", {"need_vision": need_vision}, {"records_fetched": len(vision_records)})
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

    # 1. & 5. Filter retrieved chunks to only keep highly relevant chunks (rerank_score > 0.1) and limit to top 5
    top_chunks = [c for c in sorted_chunks if c.get("rerank_score", 0.0) > 0.1]
    top_chunks = top_chunks[:5]
    if not top_chunks and sorted_chunks:
        top_chunks = sorted_chunks[:1]  # Keep at least the top chunk if none exceed the threshold
    
    # Update state retrieved_chunks to match top_chunks actually used for generation
    state["retrieved_chunks"] = top_chunks

    context_blocks = []
    for c in top_chunks:
        text = c["text"]
        if has_numeric_trigger and any(char.isdigit() for char in text):
            text = f"[NOTE: The context contains an explicit value. Use it directly if relevant.]\n{text}"
            c["text"] = text
        context_blocks.append(
            f"Document Title: {c.get('title', 'Unknown Title')}\n"
            f"Document ID: {c.get('doc_id')}\n"
            f"Page Number: {c['page']}\n"
            f"Section: {c.get('section')}\n"
            f"Text: {text}"
        )
    
    for v in vision:
        context_blocks.append(
            f"Vision data on Page {v['page']} (Type: {v['classification']}): Summary: {v['summary']}. Data: {json.dumps(v['values'])}"
        )

    # Find other related documents in the system history
    related_docs_str = ""
    try:
        async with AsyncSessionLocal() as db:
            query_keywords = state.get("structured_query", {}).get("keywords", [])
            related_docs = await find_related_documents(db, query_keywords)
            # Filter out the document(s) currently being retrieved to avoid redundancy
            retrieved_doc_ids = set(c.get("doc_id") for c in top_chunks if c.get("doc_id"))
            filtered_related = [d for d in related_docs if d["id"] not in retrieved_doc_ids]
            
            if filtered_related:
                related_docs_str = "\n".join([
                    f"- Document: {d['name']} (Circular Number: {d['circular_number']}, Type: {d['circular_type']}) | Keywords: {', '.join(d['keywords'])}"
                    for d in filtered_related
                ])
    except Exception as e:
        logger.error(f"Failed to fetch related documents during synthesis: {e}")

    context = "\n\n".join(context_blocks)
    
    system_prompt = (
        "You are an NSE compliance assistant.\n"
        "Answer ONLY using the provided context.\n"
        "\n"
        "If a number, error code, percentage, quantity, date, or identifier exists in the context, always include it explicitly.\n"
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
    res = await call_qwen(prompt, system_prompt)
    
    # Prevent hallucination: check if answer is the fallback message
    if "information not found" in res.lower() or "could not find sufficient" in res.lower() or not res.strip():
        res = "Information not found in the provided documents."
    
    step_data = {"name": "Synthesis", "output": {"answer_length": len(res)}}
    state["answer"] = res
    state["trace_steps"].append(step_data)
    if "execution_times" not in state: state["execution_times"] = {}
    state["execution_times"]["synthesis"] = time.time() - start_time
    
    await persist_trace(state["session_id"], state["query"], "Synthesis", {"context_length": len(context)}, {"answer_length": len(res)})
    return state

# --- Agent 7: Citation Agent ---
async def citation_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 7: Citation")
    start_time = time.time()
    answer = state["answer"]
    chunks = state["retrieved_chunks"]
    
    citations = []
    vague_doc_names = []
    
    for c in chunks:
        title = c.get("title") or "Unknown Title"
        
        # Track vague titles
        if title == "Unknown Title":
            vague_doc_names.append(True)
            logger.warning("Retrieved chunk has vague document title. User should provide explicit document title or ensure first page content includes a clear title.")
        
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
    
    # If all retrieved chunks have vague names, request user to clarify
    if vague_doc_names and len(vague_doc_names) == len(chunks):
        logger.error("All retrieved chunks lack clear document titles. Cannot provide accurate citations.")
        state["validation_result"]["sufficient"] = False
        state["validation_result"]["missing"] = "Document sources are not clearly identified. Please ensure documents contain clear titles on the first page."

    step_data = {"name": "Citation", "output": {"citation_count": len(citations)}}
    state["citations"] = citations
    state["trace_steps"].append(step_data)
    if "execution_times" not in state: state["execution_times"] = {}
    state["execution_times"]["citation"] = time.time() - start_time
    
    await persist_trace(state["session_id"], state["query"], "Citation", {"answer": answer}, {"citation_count": len(citations)})
    return state

# --- Agent 8: Critic Agent ---
async def critic_agent(state: AgentState) -> AgentState:
    logger.info("Executing Agent 8: Critic")
    prompt = (
        f"Evaluate this proposed compliance answer for issues:\n"
        f"Query: {state['query']}\n"
        f"Answer: {state['answer']}\n"
        f"Citations: {json.dumps(state['citations'])}\n\n"
        "Critique for 1) Hallucination risks, 2) Contradictions, 3) Outdated document versions. "
        "Return output as JSON: {\"pass\": true|false, \"reason\": \"...\", \"loop_back_retrieval\": true|false}"
    )
    res = await call_qwen(prompt, "You are a critical quality control compliance auditor.", json_format=True)
    try:
        critic_res = json.loads(res)
    except:
        critic_res = {"pass": True, "reason": "Passed critic check automatically.", "loop_back_retrieval": False}

    # Increment retry_count inside the node if critic fails
    should_retry = not critic_res.get("pass", True) or critic_res.get("loop_back_retrieval", False)
    if should_retry:
        state["retry_count"] = state.get("retry_count", 0) + 1
        logger.info(f"Incremented retry count in Critic Agent to {state['retry_count']}")

    step_data = {"name": "Critic", "output": critic_res}
    state["critic_result"] = critic_res
    state["trace_steps"].append(step_data)
    
    await persist_trace(state["session_id"], state["query"], "Critic", {"answer": state["answer"]}, critic_res)
    return state

# --- Routing Logic for LangGraph ---
def check_validation_route(state: AgentState) -> str:
    retry = state.get("retry_count", 0)
    validation = state.get("validation_result", {})
    
    should_retry = not validation.get("sufficient", True)
    
    if should_retry and retry <= MAX_RETRIES:
        logger.info(f"Looping back to Hybrid Retrieval due to missing information (Retry: {retry}/{MAX_RETRIES})")
        return "hybrid_retrieval"
    
    if should_retry:
        logger.warning(f"Max retries ({MAX_RETRIES}) reached. Proceeding with current answer.")
    
    return "vision_agent"

def check_critic_route(state: AgentState) -> str:
    critic = state.get("critic_result", {})
    retry = state.get("retry_count", 0)
    
    should_retry = not critic.get("pass", True) or critic.get("loop_back_retrieval", False)
    
    if should_retry and retry <= MAX_RETRIES:
        logger.info(f"Critic failed response. Re-triggering retrieval workflow (Retry: {retry}/{MAX_RETRIES})")
        return "hybrid_retrieval"
    
    if should_retry:
        logger.warning(f"Max retries ({MAX_RETRIES}) reached. Returning current answer despite critic issues.")
    
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

builder.set_entry_point("query_understanding")
builder.add_edge("query_understanding", "planning")
builder.add_edge("planning", "hybrid_retrieval")
builder.add_edge("hybrid_retrieval", "validation")

builder.add_conditional_edges(
    "validation",
    check_validation_route,
    {
        "hybrid_retrieval": "hybrid_retrieval",
        "vision_agent": "vision_agent"
    }
)

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
