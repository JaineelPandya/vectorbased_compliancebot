import pytest
from backend.app.services.pdf_processor import pdf_processor
from backend.app.services.reranker import CrossEncoderReranker
from backend.app.services.agent_workflow import (
    validation_agent,
    synthesis_agent,
    AgentState,
    find_related_documents,
    check_validation_route,
    check_critic_route
)
from backend.app.models import Document

def test_chunking_strategy():
    """Test Phase 2: RecursiveCharacterTextSplitter"""
    text = "word " * 350
    chunks = pdf_processor.chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 3

def test_reranker_sorting():
    """Test Phase 5: Reranker"""
    # Mocking reranker without downloading large models during test
    reranker = CrossEncoderReranker()
    # If model is loaded or simulated
    chunks = [
        {"text": "Margin verification is required for F&O.", "score": 0.9},
        {"text": "Order types in Currency Derivatives include Market, Limit, and Stop Loss.", "score": 0.8}
    ]
    query = "Explain order types"
    
    class MockModel:
        def predict(self, pairs, **kwargs):
            scores = []
            for q, c in pairs:
                if "order types" in c.lower():
                    scores.append(0.99)
                else:
                    scores.append(0.1)
            return scores
    reranker._model = MockModel()

    reranked = reranker.rerank(query, chunks, top_k=2)
    assert len(reranked) == 2
    # The second chunk should now be first because of the MockModel score
    assert "Market, Limit, and Stop Loss" in reranked[0]["text"]

@pytest.mark.asyncio
async def test_validation_mismatch_detection():
    """Test Phase 7: Validation Agent detects keyword mismatch."""
    state = AgentState(
        query="Explain order types in Currency Derivatives",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[
            {"text": "Margin verification is strictly required.", "page": 1, "rerank_score": 0.1}
        ],
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
    
    # Run validation agent
    new_state = await validation_agent(state)
    
    # Since query is about "order types" and chunk is "margin verification", mismatch should occur
    assert new_state["validation_result"]["sufficient"] is False

@pytest.mark.asyncio
async def test_synthesis_prompt_grounding():
    """Test Phase 6 & 9: Prompt grounding and citations."""
    state = AgentState(
        query="order types",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[
            {"text": "Limit orders allow users to set a specific price.", "page": 110, "title": "NSE_Currency.pdf", "document_name": "NSE_Currency.pdf", "section": "Order Types"}
        ],
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
    
    new_state = await synthesis_agent(state)
    prompt = new_state["final_prompt"]
    
    # Ensure source metadata is present in the prompt
    assert "Document Title: NSE_Currency.pdf" in prompt
    assert "Page Number: 110" in prompt
    
    # Verify strict SYSTEM prompt requirements are passed to LLM (checked by code review, the final_prompt itself doesn't contain SYSTEM prompt directly but it's part of the API call)
    assert "1. Detailed answer" in prompt
    assert "3. Source citations" in prompt

@pytest.mark.asyncio
async def test_metadata_extraction_logic():
    """Test dynamic metadata extraction fallback parser."""
    text = "The National Stock Exchange of India (NSE) issued guidelines on margin collection for currency derivatives."
    meta = pdf_processor._simulate_metadata_extraction(text)
    
    assert "margin" in meta["keywords"]
    assert "derivatives" in meta["keywords"]
    assert "SEBI" not in meta["entities"]
    assert "NSE" in meta["entities"]
    assert "margin collection" in meta["financial_terms"]

@pytest.mark.asyncio
async def test_cross_document_matching_logic():
    """Test find_related_documents matches by keywords, topics, entities overlap."""
    class MockResult:
        def __init__(self, scalar_list):
            self.scalar_list = scalar_list
        def scalars(self):
            class MockScalars:
                def __init__(self, lst):
                    self.lst = lst
                def all(self):
                    return self.lst
            return MockScalars(self.scalar_list)

    class MockSession:
        def __init__(self, docs):
            self.docs = docs
        async def execute(self, stmt):
            return MockResult(self.docs)

    doc1 = Document(
        id="11111111-2222-3333-4444-555555555555",
        name="NSE Currency Derivatives Circular",
        status="active",
        keywords=["currency", "derivatives", "margin"],
        topics=["trading segment"],
        entities=["NSE"],
        financial_terms=["margin collection"]
    )
    doc2 = Document(
        id="66666666-7777-8888-9999-000000000000",
        name="SEBI Margin Settlement Guideline",
        status="active",
        keywords=["margin", "settlement"],
        topics=["compliance regulations"],
        entities=["SEBI"],
        financial_terms=["settlement cycle"]
    )
    
    db = MockSession([doc1, doc2])
    
    res1 = await find_related_documents(db, ["currency"])
    assert len(res1) == 1
    assert res1[0]["name"] == "NSE Currency Derivatives Circular"
    
    res2 = await find_related_documents(db, ["margin"])
    assert len(res2) == 2

def test_check_validation_route_no_loop_on_prior_critic_failure():
    """Verify that if validation passes, check_validation_route does not loop back, even if a prior critic run requested loopback."""
    state = AgentState(
        query="test query",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[],
        vision_results=[],
        validation_result={"sufficient": True, "relevant": True},
        answer="",
        citations=[],
        critic_result={"pass": False, "loop_back_retrieval": True},
        retry_count=1,
        trace_steps=[],
        execution_times={},
        reranker_scores=[],
        final_prompt=""
    )
    # Even if critic result in state says loop_back_retrieval=True, we proceed to vision_agent because validation is sufficient.
    route = check_validation_route(state)
    assert route == "vision_agent"

def test_check_validation_route_loops_on_insufficient_validation():
    """Verify check_validation_route loops back when validation is insufficient and retry limit not exceeded."""
    state_retry_allowed = AgentState(
        query="test query",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[],
        vision_results=[],
        validation_result={"sufficient": False, "relevant": False},
        answer="",
        citations=[],
        critic_result={},
        retry_count=1,  # MAX_RETRIES is 1
        trace_steps=[],
        execution_times={},
        reranker_scores=[],
        final_prompt=""
    )
    # retry_count <= MAX_RETRIES (1 <= 1) -> loops back
    assert check_validation_route(state_retry_allowed) == "hybrid_retrieval"

    state_retry_exceeded = AgentState(
        query="test query",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[],
        vision_results=[],
        validation_result={"sufficient": False, "relevant": False},
        answer="",
        citations=[],
        critic_result={},
        retry_count=2,  # MAX_RETRIES is 1
        trace_steps=[],
        execution_times={},
        reranker_scores=[],
        final_prompt=""
    )
    # retry_count > MAX_RETRIES (2 > 1) -> proceeds to vision_agent
    assert check_validation_route(state_retry_exceeded) == "vision_agent"

def test_check_critic_route_loops_on_critic_failure():
    """Verify check_critic_route loops back when critic fails and retry limit not exceeded."""
    state_retry_allowed = AgentState(
        query="test query",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[],
        vision_results=[],
        validation_result={},
        answer="",
        citations=[],
        critic_result={"pass": False, "loop_back_retrieval": True},
        retry_count=1,  # MAX_RETRIES is 1
        trace_steps=[],
        execution_times={},
        reranker_scores=[],
        final_prompt=""
    )
    # retry_count <= MAX_RETRIES (1 <= 1) -> loops back
    assert check_critic_route(state_retry_allowed) == "hybrid_retrieval"

    state_retry_exceeded = AgentState(
        query="test query",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[],
        vision_results=[],
        validation_result={},
        answer="",
        citations=[],
        critic_result={"pass": False, "loop_back_retrieval": True},
        retry_count=2,  # MAX_RETRIES is 1
        trace_steps=[],
        execution_times={},
        reranker_scores=[],
        final_prompt=""
    )
    # retry_count > MAX_RETRIES (2 > 1) -> ends
    assert check_critic_route(state_retry_exceeded) == "end"

@pytest.mark.asyncio
async def test_synthesis_numeric_fact_detector_and_prioritization():
    """Verify that synthesis_agent prioritizes numeric chunks when query has keyword trigger, and injects notes."""
    state = AgentState(
        query="What error code is generated?",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[
            {"text": "No numbers here.", "page": 1, "title": "A.pdf", "rerank_score": 0.9, "doc_id": "1"},
            {"text": "Error 16448 occurred.", "page": 2, "title": "B.pdf", "rerank_score": 0.4, "doc_id": "2"}
        ],
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

    new_state = await synthesis_agent(state)
    
    # Check that chunks used in generation are sorted so the numeric one is first (due to query keyword 'error code')
    used_chunks = new_state["retrieved_chunks"]
    assert len(used_chunks) == 2
    assert "Error 16448" in used_chunks[0]["text"]
    assert "[NOTE: The context contains an explicit value. Use it directly if relevant.]" in used_chunks[0]["text"]
    assert "[NOTE: The context contains" not in used_chunks[1]["text"]

@pytest.mark.asyncio
async def test_synthesis_rerank_filtering():
    """Verify that synthesis_agent filters out chunks with rerank_score <= 0.1, keeping top 5, or keeping top 1 as fallback."""
    state = AgentState(
        query="Explain compliance rules",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[
            {"text": "Relevant chunk.", "page": 1, "title": "A.pdf", "rerank_score": 0.8, "doc_id": "1"},
            {"text": "Low score chunk.", "page": 2, "title": "B.pdf", "rerank_score": 0.05, "doc_id": "2"},
            {"text": "Another low score.", "page": 3, "title": "C.pdf", "rerank_score": 0.02, "doc_id": "3"}
        ],
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

    new_state = await synthesis_agent(state)
    
    # Low score chunks should be filtered out, leaving only the one with score 0.8
    used_chunks = new_state["retrieved_chunks"]
    assert len(used_chunks) == 1
    assert "Relevant chunk." in used_chunks[0]["text"]

@pytest.mark.asyncio
async def test_synthesis_rerank_filtering_fallback():
    """Verify that synthesis_agent keeps at least the top chunk if all chunks have rerank_score <= 0.1."""
    state = AgentState(
        query="Explain compliance rules",
        session_id="test",
        filters={},
        structured_query={},
        search_tasks=[],
        retrieved_chunks=[
            {"text": "Low score chunk.", "page": 2, "title": "B.pdf", "rerank_score": 0.05, "doc_id": "2"},
            {"text": "Another low score.", "page": 3, "title": "C.pdf", "rerank_score": 0.02, "doc_id": "3"}
        ],
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

    new_state = await synthesis_agent(state)
    
    # Should fall back to keeping the top chunk (0.05 score) rather than returning empty list
    used_chunks = new_state["retrieved_chunks"]
    assert len(used_chunks) == 1
    assert "Low score chunk." in used_chunks[0]["text"]
