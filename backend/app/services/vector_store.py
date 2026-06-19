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
        self.collection = "document_chunks"
        self.vector_size = 1024  # Matches BGE-M3 embedding dimension
        self.connect()

    def connect(self):
        try:
            # Setup standard Qdrant Client
            self.client = QdrantClient(url=self.url)
            self._init_collection()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.url}: {e}. Local mock mode may be triggered during execution.")

    def _init_collection(self):
        if not self.client:
            return
        
        try:
            self.client.get_collection(collection_name=self.collection)
            logger.info(f"Qdrant collection '{self.collection}' already exists.")
        except (UnexpectedResponse, Exception):
            logger.info(f"Creating Qdrant collection '{self.collection}'...")
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE
                    )
                )
                logger.info(f"Successfully created collection '{self.collection}'.")
            except Exception as e:
                logger.error(f"Error creating collection '{self.collection}': {e}")

    async def upsert_chunks(
        self, 
        points: List[Dict[str, Any]]
    ) -> bool:
        """
        Upserts vectors and metadata payload into the document_chunks Qdrant collection.
        points is a list of dicts:
        {
           "id": uuid,
           "vector": list[float],
           "payload": {
               "title": str,
               "doc_id": str,
               "page": int,
               "section": str,
               "subsection": str,
               "circular_number": str,
               "document_version": int,
               "type": str,
               "text": str
           }
        }
        """
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
                collection_name=self.collection,
                points=q_points
            )
            logger.info(f"Upserted {len(points)} vectors into {self.collection}.")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {e}")
            return False

    async def search_collection(
        self,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the document_chunks collection for similar vectors.
        Supports filtering on metadata fields.
        """
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
                collection_name=self.collection,
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
        """Deletes all vectors associated with a document from the document_chunks collection."""
        if not self.client:
            return True

        try:
            self.client.delete(
                collection_name=self.collection,
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
            logger.info(f"Deleted vectors for doc_id {doc_id} from {self.collection}.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete Qdrant vectors for doc_id {doc_id}: {e}")
            return False
qdrant_store = QdrantStore()
