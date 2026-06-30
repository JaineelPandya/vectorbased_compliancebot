import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Date, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from backend.app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    circular_number = Column(String, nullable=True, index=True)
    hash = Column(String, unique=True, nullable=False, index=True)
    version = Column(Integer, default=1, nullable=False)
    status = Column(String, default="processing", nullable=False)  # active, archived, processing, failed
    issue_date = Column(Date, nullable=True, index=True)
    department = Column(String, nullable=True, index=True)
    tags = Column(JSON, nullable=True)  # List of strings e.g., ["margin", "f&o"]
    keywords = Column(JSON, nullable=True)
    topics = Column(JSON, nullable=True)
    entities = Column(JSON, nullable=True)
    financial_terms = Column(JSON, nullable=True)
    circular_type = Column(String, nullable=True)
    subject = Column(Text, nullable=True)   # P0: subject line extracted from first page
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    pages = relationship("DocumentPage", back_populates="document", cascade="all, delete-orphan")
    logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")

class DocumentPage(Base):
    __tablename__ = "document_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    page_number = Column(Integer, nullable=False)
    classification = Column(String, nullable=False)  # text, scanned, graph, table, mixed
    confidence = Column(Float, default=1.0, nullable=False)
    vision_summary = Column(Text, nullable=True)
    vision_extracted_values = Column(JSON, nullable=True)  # e.g., {"revenue": 30000000}

    document = relationship("Document", back_populates="pages")

class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    step = Column(String, nullable=False)  # UPLOAD, CLASSIFY, CHUNK, EMBED, QDRANT, ES, COMPLETE
    status = Column(String, nullable=False)  # SUCCESS, FAILED, RUNNING
    message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="logs")

class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    query = Column(Text, nullable=False)
    step_name = Column(String, nullable=False)  # Query Understanding, Planning, Hybrid Retrieval, etc.
    input_state = Column(JSON, nullable=True)
    output_state = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
