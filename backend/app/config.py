import os
import pathlib
import yaml
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Base Directory of the Project
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"

class AppConfig(BaseModel):
    name: str = "Agentic PDF Search System"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

class DatabaseConfig(BaseModel):
    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/pdf_search"
    sync_url: str = "postgresql://postgres:postgres@localhost:5432/pdf_search"

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    prefer_grpc: bool = False

class ElasticsearchConfig(BaseModel):
    hosts: List[str] = ["http://localhost:9200"]

class EmbeddingsConfig(BaseModel):
    model_name: str = "BAAI/bge-m3"
    use_ollama: bool = True
    ollama_url: str = "http://localhost:11434"

class LLMConfig(BaseModel):
    reasoning_model: str = "qwen3:latest"
    vision_model: str = "qwen3-vl:latest"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.3 

class StorageConfig(BaseModel):
    upload_dir: str = "./storage/uploads"
    temp_dir: str = "./storage/temp"
    max_file_size_mb: int = 50

class ChunkingConfig(BaseModel):
    # TEXT pages — sliding window
    text_chunk_size: int = 600        # Max characters per sliding window
    text_overlap_sentences: int = 2   # Sentences to carry into next window
    text_min_chunk_len: int = 30      # Minimum chars to keep a window
    # TABLE pages — row-based split
    table_max_rows: int = 20          # Max table rows per chunk
    # GRAPH pages — always single chunk via vision pipeline

class SystemSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)

def load_settings() -> SystemSettings:
    """Loads system settings from YAML config and env vars."""
    config_dict: Dict[str, Any] = {}
    
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            try:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml:
                    config_dict = loaded_yaml
            except Exception as e:
                print(f"Warning: Failed to parse {CONFIG_PATH}: {e}. Using defaults.")
    
    # Allow environment variable overrides (e.g. DB_URL)
    if os.getenv("DATABASE_URL"):
        config_dict.setdefault("database", {})["url"] = os.getenv("DATABASE_URL")
    if os.getenv("REDIS_HOST"):
        config_dict.setdefault("redis", {})["host"] = os.getenv("REDIS_HOST")
    if os.getenv("QDRANT_URL"):
        config_dict.setdefault("qdrant", {})["url"] = os.getenv("QDRANT_URL")
    if os.getenv("ELASTICSEARCH_HOSTS"):
        hosts = os.getenv("ELASTICSEARCH_HOSTS", "").split(",")
        config_dict.setdefault("elasticsearch", {})["hosts"] = [h.strip() for h in hosts if h.strip()]
    if os.getenv("OLLAMA_URL"):
        config_dict.setdefault("llm", {})["ollama_url"] = os.getenv("OLLAMA_URL")
        config_dict.setdefault("embeddings", {})["ollama_url"] = os.getenv("OLLAMA_URL")

    settings = SystemSettings(**config_dict)
    
    # Ensure directories exist
    pathlib.Path(settings.storage.upload_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(settings.storage.temp_dir).mkdir(parents=True, exist_ok=True)
    
    return settings

settings = load_settings()
