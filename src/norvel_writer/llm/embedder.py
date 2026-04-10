"""Batched embedding service with retry/backoff (LangChain backend)."""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

BATCH_SIZE = 32
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Infer the embedding vector dimension from the configured model so that
# fallback zero-vectors always match the active collection's dimension.
# Only used on hard failure (model not found, service down, etc.).
def _infer_zero_dim() -> int:
    import os
    model = os.environ.get("OLLAMA_EMBED_MODEL", "").lower()
    # 1024-dim models
    if any(k in model for k in ("bge", "mxbai", "snowflake-arctic")):
        return 1024
    # OpenAI models
    oai = os.environ.get("OPENAI_EMBED_MODEL", "").lower()
    if "3-large" in oai:
        return 3072
    if "3-small" in oai or "ada" in oai:
        return 1536
    # Default: nomic-embed-text and most small Ollama models = 768
    return 768

_ZERO_DIM = _infer_zero_dim()

# bge-m3 context limit: 8 192 tokens ≈ 32 768 chars
#
# _EMBED_DOC_MAX_CHARS — hard ceiling applied to every chunk before it is sent
#   to the embedding API.  The chunker targets 512 tokens ≈ 2 048 chars, but
#   LangChain's MarkdownTextSplitter treats chunk_size as a *soft* limit and
#   will emit oversized blocks when no internal split point exists (common with
#   dense CJK prose or codex entries with no blank lines).  Truncating here
#   prevents HTTP 400 "input length exceeds context length" errors.
#   8 000 chars ≈ 2 000 tokens — safely under bge-m3's 8 192-token limit.
#   Normal 512-token chunks are ≈ 2 048 chars, well below this ceiling.
_EMBED_DOC_MAX_CHARS   = 8000   # ≈ 2 000 tokens — document chunks
_EMBED_QUERY_MAX_CHARS = 2000   # ≈   500 tokens — RAG queries (less = fine)


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
            batch = [
                t[:_EMBED_DOC_MAX_CHARS] if len(t) > _EMBED_DOC_MAX_CHARS else t
                for t in texts[batch_start : batch_start + BATCH_SIZE]
            ]
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
        # Truncate long inputs — embedding models have a context limit and a
        # long passage doesn't improve query quality over a short representative
        # excerpt.  Prevents HTTP 400 "input length exceeds context length".
        if len(text) > _EMBED_QUERY_MAX_CHARS:
            log.debug(
                "embed_single: truncating query from %d → %d chars",
                len(text), _EMBED_QUERY_MAX_CHARS,
            )
            text = text[:_EMBED_QUERY_MAX_CHARS]
        results = await self.embed_texts([text])
        return results[0] if results else []
