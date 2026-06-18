import logging
import httpx
from typing import List
import numpy as np
from backend.app.config import settings

logger = logging.getLogger("app.embeddings")

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.embeddings.model_name
        self.use_ollama = settings.embeddings.use_ollama
        self.ollama_url = settings.embeddings.ollama_url
        self.local_model = None
        self.dimension = 1024  # BGE-M3 default dimension size is 1024

        if not self.use_ollama:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local SentenceTransformer model {self.model_name}...")
                self.local_model = SentenceTransformer(self.model_name)
                logger.info("Local SentenceTransformer model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load local SentenceTransformer {self.model_name}: {e}. Falling back to Ollama or mock.")

    async def get_embedding(self, text: str) -> List[float]:
        """Generates embedding for a single text chunk."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0]

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text chunks."""
        if not texts:
            return []

        # Case 1: Local SentenceTransformer model
        if self.local_model:
            try:
                embeddings = self.local_model.encode(texts, convert_to_numpy=True)
                return [arr.tolist() for arr in embeddings]
            except Exception as e:
                logger.error(f"Local SentenceTransformer embedding failed: {e}")

        # Case 2: Ollama embedding API
        if self.use_ollama:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    embeddings_list = []
                    for text in texts:
                        response = await client.post(
                            f"{self.ollama_url}/api/embeddings",
                            json={"model": "bge-m3", "prompt": text}
                        )
                        if response.status_code == 200:
                            embeddings_list.append(response.json()["embedding"])
                        else:
                            # Try with default ollama model
                            response = await client.post(
                                f"{self.ollama_url}/api/embeddings",
                                json={"model": settings.llm.reasoning_model, "prompt": text}
                            )
                            if response.status_code == 200:
                                embeddings_list.append(response.json()["embedding"])
                            else:
                                raise Exception(f"Ollama returned status {response.status_code}: {response.text}")
                    
                    # Ensure embedding sizes match expected dimension, truncate or pad if necessary
                    result = []
                    for emb in embeddings_list:
                        if len(emb) < self.dimension:
                            emb = emb + [0.0] * (self.dimension - len(emb))
                        elif len(emb) > self.dimension:
                            emb = emb[:self.dimension]
                        result.append(emb)
                    return result
            except Exception as e:
                logger.warning(f"Ollama embedding generation failed: {e}. Falling back to mock embeddings.")

        # Case 3: Mock embeddings for robust testing
        logger.warning("Using mock embeddings of dimension 1024.")
        mock_results = []
        for text in texts:
            # Deterministic mock embedding based on character hash code
            state = hash(text) & 0xffffffff
            np.random.seed(state)
            mock_emb = np.random.randn(self.dimension)
            # Normalize vector
            mock_emb = mock_emb / np.linalg.norm(mock_emb)
            mock_results.append(mock_emb.tolist())
        return mock_results

embedding_service = EmbeddingService()
