import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from backend.app.config import settings

logger = logging.getLogger("app.vector_store")

class QdrantStore:
    def __init__(self):
        self.url = settings.qdrant.url
        self.client = None
        self.collections = ["text_chunks", "table_chunks", "graph_chunks", "scan_chunks"]
        self.vector_size = 1024  # Matches BGE-M3 embedding dimension
        self.connect()

    def connect(self):
        try:
            # Setup standard Qdrant Client
            self.client = QdrantClient(url=self.url)
            self._init_collections()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.url}: {e}. Local mock mode may be triggered during execution.")

    def _init_collections(self):
        if not self.client:
            return
        
        for collection in self.collections:
            try:
                # Check if collection exists
                self.client.get_collection(collection_name=collection)
                logger.info(f"Qdrant collection '{collection}' already exists.")
            except (UnexpectedResponse, Exception):
                logger.info(f"Creating Qdrant collection '{collection}'...")
                try:
                    self.client.create_collection(
                        collection_name=collection,
                        vectors_config=qmodels.VectorParams(
                            size=self.vector_size,
                            distance=qmodels.Distance.COSINE
                        )
                    )
                    logger.info(f"Successfully created collection '{collection}'.")
                except Exception as e:
                    logger.error(f"Error creating collection '{collection}': {e}")

    async def upsert_chunks(
        self, 
        collection: str, 
        points: List[Dict[str, Any]]
    ) -> bool:
        """
        Upserts vectors and metadata payload into a Qdrant collection.
        points is a list of dicts:
        {
           "id": uuid,
           "vector": list[float],
           "payload": {
               "doc_id": str,
               "page": int,
               "section": str,
               "document_version": int,
               "upload_date": str,
               "text": str
           }
        }
        """
        if collection not in self.collections:
            logger.error(f"Invalid Qdrant collection target: {collection}")
            return False

        if not self.client:
            logger.warning("Qdrant client not connected. Simulating success.")
            return True

        try:
            q_points = []
            for p in points:
                q_points.append(
                    qmodels.PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p["payload"]
                    )
                )
            
            self.client.upsert(
                collection_name=collection,
                points=q_points
            )
            logger.info(f"Upserted {len(points)} vectors into {collection}.")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {e}")
            return False

    async def search_collection(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches a Qdrant collection for similar vectors.
        Supports filtering on metadata fields.
        """
        if collection not in self.collections:
            logger.error(f"Invalid Qdrant collection target: {collection}")
            return []

        if not self.client:
            logger.warning("Qdrant client not connected. Returning empty search results.")
            return []

        try:
            q_filter = None
            if filters:
                must_conditions = []
                for k, v in filters.items():
                    if v is not None:
                        must_conditions.append(
                            qmodels.FieldCondition(
                                key=k,
                                match=qmodels.MatchValue(value=v)
                            )
                        )
                if must_conditions:
                    q_filter = qmodels.Filter(must=must_conditions)

            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                query_filter=q_filter,
                with_payload=True
                )

            results = response.points
            ret = []
            for r in results:
                ret.append({
                    "id": r.id,
                    "score": r.score,
                    "payload": r.payload
                })
            return ret
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    async def delete_document_vectors(self, doc_id: str) -> bool:
        """Deletes all vectors associated with a document across all collections."""
        if not self.client:
            return True

        try:
            for collection in self.collections:
                self.client.delete(
                    collection_name=collection,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            must=[
                                qmodels.FieldCondition(
                                    key="doc_id",
                                    match=qmodels.MatchValue(value=doc_id)
                                )
                            ]
                        )
                    )
                )
            logger.info(f"Deleted vectors for doc_id {doc_id} across all collections.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Qdrant vectors for doc_id {doc_id}: {e}")
            return False

qdrant_store = QdrantStore()
