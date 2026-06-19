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
            try:
                # Use singleton pattern with MPS device and batch_size=32
                self._model = CrossEncoder(self.model_name, device="mps")
                logger.info(f"Loaded reranker model {self.model_name} with device=mps")
            except Exception as e:
                logger.exception(f"Failed to load reranker {self.model_name}: {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """Reranks a list of retrieved chunks using a CrossEncoder."""
        if not chunks:
            return []
        
        if not self.model:
            logger.warning("Reranker model not loaded. Returning original chunks.")
            return chunks[:top_k]

        # Log pairs being sent to CrossEncoder
        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
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
            return chunks[:top_k]

reranker = CrossEncoderReranker()
