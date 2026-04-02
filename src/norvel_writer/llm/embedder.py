"""Batched embedding service with retry/backoff."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_RETRIES = 3
RETRY_DELAY = 2.0


class EmbeddingService:
    def __init__(
        self,
        model: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        from norvel_writer.config.settings import get_config
        self._model = model or get_config().default_embed_model
        self._progress_cb = progress_cb

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        Returns embeddings in the same order as input.
        """
        from norvel_writer.llm.ollama_client import get_client
        client = get_client()

        all_embeddings: List[List[float]] = []
        total = len(texts)

        for batch_start in range(0, total, BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            for attempt in range(MAX_RETRIES):
                try:
                    embeddings = await client.embed(self._model, batch)
                    all_embeddings.extend(embeddings)
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES - 1:
                        log.error("Embedding batch failed after %d retries: %s", MAX_RETRIES, exc)
                        # Return zero vectors for failed batch
                        all_embeddings.extend([[0.0] * 768] * len(batch))
                    else:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))

            if self._progress_cb:
                done = min(batch_start + BATCH_SIZE, total)
                self._progress_cb(done, total)

        return all_embeddings

    async def embed_single(self, text: str) -> List[float]:
        results = await self.embed_texts([text])
        return results[0] if results else []
