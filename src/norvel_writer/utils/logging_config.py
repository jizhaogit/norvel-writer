"""Logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "norvel_writer.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Quiet noisy libraries
    for noisy in ("chromadb", "httpx", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ChromaDB's PostHog telemetry client has a broken call signature that
    # logs ERROR on every event even when telemetry is disabled.
    # Silence it completely — it carries no actionable information.
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    # ChromaDB's HNSW segment emits a WARNING when n_results > collection size.
    # This is harmless (ChromaDB clamps it automatically), but noisy — especially
    # right after a small upload when fewer chunks exist than the default n_results.
    logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)
