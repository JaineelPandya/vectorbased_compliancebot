from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import uuid

# --- Document Schemas ---
class DocumentBase(BaseModel):
    name: str
    circular_number: Optional[str] = None
    issue_date: Optional[date] = None
    department: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)

class DocumentCreate(DocumentBase):
    pass

class DocumentResponse(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    hash: str
    version: int
    status: str
    uploaded_at: datetime

# --- Document Page Schemas ---
class PageClassificationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_number: int
    classification: str
    confidence: float
    vision_summary: Optional[str] = None
    vision_extracted_values: Optional[Dict[str, Any]] = None

# --- Ingestion Log Schemas ---
class ProcessingLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    step: str
    status: str
    message: Optional[str] = None
    timestamp: datetime

# --- Agentic Search / Query Schemas ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    filters: Optional[Dict[str, Any]] = None  # e.g., {"department": "SEBI", "start_date": "2026-01-01"}

class Citation(BaseModel):
    document_name: str
    page_number: int
    section: Optional[str] = None
    circular_number: Optional[str] = None
    version: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    agent_trace_session_id: str

# --- Agent Trace Schemas ---
class AgentTraceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    query: str
    step_name: str
    input_state: Optional[Dict[str, Any]] = None
    output_state: Optional[Dict[str, Any]] = None
    timestamp: datetime
