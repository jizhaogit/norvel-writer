"""ChromaDB wrapper for local vector storage."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class QueryResult:
    def __init__(self, id: str, text: str, distance: float, metadata: Dict[str, Any]) -> None:
        self.id = id
        self.text = text
        self.distance = distance
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"QueryResult(id={self.id!r}, distance={self.distance:.4f})"


class VectorStore:
    """Manages ChromaDB persistent collections for projects and style profiles."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        import chromadb
        # Disable ChromaDB's PostHog telemetry — their integration has a bug
        # ("capture() takes 1 positional argument but 3 were given") that spams
        # the log on every collection event.
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._dim_cache: Dict[str, int] = {}

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self, name: str) -> bool:
        try:
            self._client.get_collection(name)
            return True
        except Exception:
            return False

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
            self._dim_cache.pop(name, None)
        except Exception:
            pass

    def upsert_chunks(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        col = self._collection(collection_name)
        # Validate dimension consistency
        if embeddings and embeddings[0]:
            dim = len(embeddings[0])
            if collection_name in self._dim_cache:
                if self._dim_cache[collection_name] != dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for collection {collection_name!r}: "
                        f"expected {self._dim_cache[collection_name]}, got {dim}"
                    )
            else:
                self._dim_cache[collection_name] = dim

        # Sanitize metadata — ChromaDB rejects None values
        clean_meta = [
            {k: v for k, v in m.items() if v is not None} for m in metadatas
        ]
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_meta,
        )

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.collection_exists(collection_name):
            return []
        col = self._collection(collection_name)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        try:
            results = col.query(**kwargs)
        except Exception as exc:
            log.warning("Vector query failed: %s", exc)
            return []

        out: List[QueryResult] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i, rid in enumerate(ids):
            out.append(
                QueryResult(
                    id=rid,
                    text=docs[i] if docs else "",
                    distance=dists[i] if dists else 0.0,
                    metadata=metas[i] if metas else {},
                )
            )
        return out

    def delete_by_document(self, collection_name: str, document_id: str) -> None:
        if not self.collection_exists(collection_name):
            return
        col = self._collection(collection_name)
        try:
            col.delete(where={"document_id": document_id})
        except Exception as exc:
            log.warning("Failed to delete chunks for document %s: %s", document_id, exc)


_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        from norvel_writer.config.settings import get_config
        _store = VectorStore(get_config().chroma_path)
    return _store


def init_vector_store(path: Path) -> VectorStore:
    global _store
    _store = VectorStore(path)
    return _store
