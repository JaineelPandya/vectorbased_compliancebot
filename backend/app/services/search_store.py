import logging
from typing import List, Dict, Any, Optional
from elasticsearch import AsyncElasticsearch
from backend.app.config import settings

logger = logging.getLogger("app.search_store")

class ElasticsearchStore:
    def __init__(self):
        self.hosts = settings.elasticsearch.hosts
        self.client = None
        self.doc_index = "documents_metadata"
        self.chunk_index = "document_chunks"
        self.connect()

    def connect(self):
        try:
            # Setup standard Async Elasticsearch client
            self.client = AsyncElasticsearch(hosts=self.hosts)
            logger.info(f"Connected to Elasticsearch at {self.hosts}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")

    async def init_indices(self):
        """Creates indexes and mappings if they do not exist."""
        if not self.client:
            return

        # 1. Document Metadata Index Mapping
        doc_mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "standard"},
                    "circular_number": {"type": "keyword"},
                    "issue_date": {"type": "date"},
                    "version": {"type": "integer"},
                    "department": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                    "topics": {"type": "keyword"},
                    "entities": {"type": "keyword"},
                    "financial_terms": {"type": "keyword"},
                    "circular_type": {"type": "keyword"},
                    "status": {"type": "keyword"},
                    "uploaded_at": {"type": "date"}
                }
            }
        }

        # 2. Document Chunks Index Mapping (for Full-Text / BM25 search)
        chunk_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "section": {"type": "text"},
                    "subsection": {"type": "text"},
                    "parent_id": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "english"},
                    "circular_number": {"type": "keyword"},
                    "issue_date": {"type": "date"},
                    "version": {"type": "integer"},
                    "department": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "keywords": {"type": "keyword"},
                    "topics": {"type": "keyword"},
                    "entities": {"type": "keyword"},
                    "financial_terms": {"type": "keyword"},
                    "circular_type": {"type": "keyword"},
                    "status": {"type": "keyword"}
                }
            }
        }

        for idx, mapping in [(self.doc_index, doc_mapping), (self.chunk_index, chunk_mapping)]:
            try:
                exists = await self.client.indices.exists(index=idx)
                if not exists:
                    await self.client.indices.create(index=idx, body=mapping)
                    logger.info(f"Created Elasticsearch index '{idx}'")
                else:
                    logger.info(f"Elasticsearch index '{idx}' already exists.")
            except Exception as e:
                logger.error(f"Failed to create index '{idx}': {e}")

    async def index_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Indexes document metadata into Elasticsearch."""
        if not self.client:
            return True

        try:
            body = {
                "doc_id": doc_id,
                "name": metadata.get("name"),
                "circular_number": metadata.get("circular_number"),
                "issue_date": metadata.get("issue_date"),
                "version": metadata.get("version", 1),
                "department": metadata.get("department"),
                "tags": metadata.get("tags", []),
                "keywords": metadata.get("keywords", []),
                "topics": metadata.get("topics", []),
                "entities": metadata.get("entities", []),
                "financial_terms": metadata.get("financial_terms", []),
                "circular_type": metadata.get("circular_type"),
                "status": metadata.get("status", "active"),
                "uploaded_at": metadata.get("uploaded_at")
            }
            await self.client.index(index=self.doc_index, id=doc_id, body=body, refresh="wait_for")
            logger.info(f"Indexed document metadata for doc_id {doc_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to index document metadata in ES: {e}")
            return False

    async def index_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Indexes multiple text chunks for full-text search."""
        if not self.client:
            return True

        try:
            for chunk in chunks:
                body = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk.get("title") or "Unknown Title",
                    "page_number": chunk.get("page") or chunk.get("page_number") or 1,
                    "section": chunk.get("section") or "Unknown Section",
                    "subsection": chunk.get("subsection") or "",
                    "parent_id": chunk.get("parent_id") or "",
                    "type": chunk.get("type") or "text",
                    "text": chunk.get("text") or "",
                    "circular_number": chunk.get("circular_number") or "",
                    "issue_date": chunk.get("issue_date"),
                    "version": chunk.get("document_version") or chunk.get("version") or 1,
                    "department": chunk.get("department") or "Unknown",
                    "tags": chunk.get("tags") or [],
                    "keywords": chunk.get("keywords") or [],
                    "topics": chunk.get("topics") or [],
                    "entities": chunk.get("entities") or [],
                    "financial_terms": chunk.get("financial_terms") or [],
                    "circular_type": chunk.get("circular_type") or "",
                    "status": chunk.get("status") or "active"
                }
                await self.client.index(index=self.chunk_index, id=chunk["chunk_id"], body=body)
            await self.client.indices.refresh(index=self.chunk_index)
            logger.info(f"Indexed {len(chunks)} chunks into Elasticsearch.")
            return True
        except Exception as e:
            logger.error(f"Failed to index chunks in ES: {e}")
            return False

    async def search_metadata(
        self,
        circular_number: Optional[str] = None,
        department: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        status: Optional[str] = None,
        version: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Searches document metadata with precise filters."""
        if not self.client:
            return []

        try:
            must = []
            if circular_number:
                must.append({"term": {"circular_number": circular_number}})
            if department:
                must.append({"term": {"department": department}})
            if status:
                must.append({"term": {"status": status}})
            if version:
                must.append({"term": {"version": version}})
            
            # Add range query for issue_date if provided
            if start_date or end_date:
                date_range = {}
                if start_date:
                    date_range["gte"] = start_date
                if end_date:
                    date_range["lte"] = end_date
                must.append({"range": {"issue_date": date_range}})

            query = {"query": {"bool": {"must": must}}} if must else {"query": {"match_all": {}}}
            
            response = await self.client.search(index=self.doc_index, body=query)
            hits = response["hits"]["hits"]
            return [hit["_source"] for hit in hits]
        except Exception as e:
            logger.error(f"Elasticsearch metadata search failed: {e}")
            return []

    async def search_chunks(
        self,
        text_query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Performs full-text keyword search (BM25) over chunks, applying filters."""
        if not self.client:
            return []

        try:
            must = [{"match": {"text": text_query}}]
            filter_queries = []
            
            if filters:
                for k, v in filters.items():
                    if v is not None:
                        if k == "start_date" or k == "end_date":
                            date_range = {}
                            if k == "start_date":
                                date_range["gte"] = v
                            if k == "end_date":
                                date_range["lte"] = v
                            filter_queries.append({"range": {"issue_date": date_range}})
                        else:
                            filter_queries.append({"term": {k: v}})

            query = {
                "query": {
                    "bool": {
                        "must": must,
                        "filter": filter_queries
                    }
                },
                "size": limit
            }

            response = await self.client.search(index=self.chunk_index, body=query)
            hits = response["hits"]["hits"]
            
            ret = []
            for hit in hits:
                source = hit["_source"]
                ret.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "payload": {
                        "doc_id": source["doc_id"],
                        "title": source.get("title"),
                        "page": source.get("page_number", source.get("page")),
                        "section": source.get("section"),
                        "subsection": source.get("subsection"),
                        "parent_id": source.get("parent_id"),
                        "type": source.get("type", "text"),
                        "text": source.get("text"),
                        "circular_number": source.get("circular_number"),
                        "document_version": source.get("version"),
                        "upload_date": source.get("issue_date"),
                        "keywords": source.get("keywords") or [],
                        "topics": source.get("topics") or [],
                        "entities": source.get("entities") or [],
                        "financial_terms": source.get("financial_terms") or [],
                        "circular_type": source.get("circular_type") or ""
                    }
                })
            return ret
        except Exception as e:
            logger.error(f"Elasticsearch full-text search failed: {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Deletes metadata and associated chunks from ES."""
        if not self.client:
            return True

        try:
            # Delete document metadata
            await self.client.delete(index=self.doc_index, id=doc_id, ignore=[404])
            # Delete chunks belonging to the document
            query = {"query": {"term": {"doc_id": doc_id}}}
            await self.client.delete_by_query(index=self.chunk_index, body=query)
            logger.info(f"Deleted ES records for doc_id {doc_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document from ES: {e}")
            return False

search_store = ElasticsearchStore()
