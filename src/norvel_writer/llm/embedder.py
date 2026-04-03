"""Batched embedding service with retry/backoff (LangChain backend)."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Dimension used for zero-vectors on failure.  Must match the embed model.
# nomic-embed-text = 768; text-embedding-3-small = 1536
_ZERO_DIM = 768


class EmbeddingService:
    def __init__(
        self,
        model: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        # model parameter is kept for API compatibility but is no longer used;
        # the active model is determined by .env (OLLAMA_EMBED_MODEL / OPENAI_EMBED_MODEL).
        self._progress_cb = progress_cb

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts in batches via the LangChain embeddings backend.
        Returns embeddings in the same order as input.
        """
        from norvel_writer.llm.langchain_bridge import get_embeddings_fn

        embeddings_fn = get_embeddings_fn()
        if embeddings_fn is None:
            log.error("Embeddings backend unavailable — returning zero vectors.")
            return [[0.0] * _ZERO_DIM] * len(texts)

        all_embeddings: List[List[float]] = []
        total = len(texts)

        for batch_start in range(0, total, BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            for attempt in range(MAX_RETRIES):
                try:
                    embeddings = await embeddings_fn.aembed_documents(batch)
                    all_embeddings.extend(embeddings)
                    break
                except Exception as exc:
                    err_str = str(exc).lower()
                    # "not found" / 404 errors are permanent — the model is not
                    # pulled.  Retrying wastes time; fail immediately.
                    is_permanent = (
                        "not found" in err_str
                        or "404" in err_str
                        or "does not exist" in err_str
                        or "no such model" in err_str
                    )
                    if attempt == MAX_RETRIES - 1 or is_permanent:
                        log.error(
                            "Embedding batch failed%s: %s",
                            " (model not found — skipping retries)" if is_permanent else f" after {MAX_RETRIES} retries",
                            exc,
                        )
                        dim = len(all_embeddings[0]) if all_embeddings else _ZERO_DIM
                        all_embeddings.extend([[0.0] * dim] * len(batch))
                        break
                    else:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))

            if self._progress_cb:
                done = min(batch_start + BATCH_SIZE, total)
                self._progress_cb(done, total)

        return all_embeddings

    async def embed_single(self, text: str) -> List[float]:
        results = await self.embed_texts([text])
        return results[0] if results else []
