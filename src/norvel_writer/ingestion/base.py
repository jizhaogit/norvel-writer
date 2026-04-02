"""Base ingestor interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class IngestedDocument:
    text: str
    title: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class BaseIngestor(ABC):
    """Extract raw text and metadata from a file."""

    @abstractmethod
    def can_handle(self, path: Path) -> bool: ...

    @abstractmethod
    def ingest(self, path: Path) -> IngestedDocument: ...
