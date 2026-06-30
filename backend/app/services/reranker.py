import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger("app.reranker")

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            for device in ["mps", "cpu"]:
                try:
                    self._model = CrossEncoder(self.model_name, device=device)
                    logger.info(f"Loaded reranker model {self.model_name} with device={device}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load reranker on device={device}: {e}")
            if self._model is None:
                logger.error(
                    f"Reranker model {self.model_name} could not be loaded on any device. "
                    "All chunks will receive rerank_score=0.0."
                )
        return self._model

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """Reranks a list of retrieved chunks using a CrossEncoder."""
        if not chunks:
            return []

        if not self.model:
            logger.warning(
                "Reranker model not loaded. Assigning rerank_score=0.0 to all chunks "
                "so downstream key access never crashes."
            )
            for chunk in chunks:
                chunk.setdefault("rerank_score", 0.0)
            return chunks[:top_k]

        # Log pairs being sent to CrossEncoder. Prefer `reranker_text` if provided
        pairs = [[query, (chunk.get("reranker_text") or chunk.get("text", ""))] for chunk in chunks]
        logger.info(f"Reranking {len(pairs)} pairs for query: {query[:60]}...")
        for idx, (q, text) in enumerate(pairs[:3]):
            page_num = chunks[idx].get("page", "N/A")
            score = chunks[idx].get("score", "N/A")
            logger.info(f"Pair {idx}: page={page_num}, similarity_score={score}, text_preview={text[:60]}...")
        
        try:
            # Use batch_size=32 for efficiency
            scores = self.model.predict(pairs, batch_size=32)
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
            
            # Sort by rerank score descending
            reranked_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", -999.0), reverse=True)
            
            # Log rerank scores
            logger.info("Reranked chunks (top 3):")
            for idx, chunk in enumerate(reranked_chunks[:3]):
                logger.info(f"  {idx+1}. rerank_score={chunk.get('rerank_score'):.4f}, page={chunk.get('page')}, title={chunk.get('title')}")
            
            return reranked_chunks[:top_k]
        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            # Inject rerank_score=0.0 so downstream .get("rerank_score", 0.0)
            # calls and direct key accesses never raise KeyError.
            for chunk in chunks:
                chunk.setdefault("rerank_score", 0.0)
            return chunks[:top_k]

reranker = CrossEncoderReranker()
