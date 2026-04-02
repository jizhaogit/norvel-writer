from __future__ import annotations
from pathlib import Path
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument
from norvel_writer.utils.text_utils import normalize_whitespace


class PdfIngestor(BaseIngestor):
    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def ingest(self, path: Path) -> IngestedDocument:
        import pdfplumber
        pages: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    pages.append(extracted)
        text = normalize_whitespace("\n\n".join(pages))
        return IngestedDocument(
            text=text,
            title=path.stem,
            metadata={"page_count": str(len(pages))},
        )
