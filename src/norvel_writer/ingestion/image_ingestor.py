"""
Image ingestor: converts images (maps, character art, etc.) to text descriptions
using a vision-capable Ollama model, then stores those descriptions in the
knowledge base for RAG retrieval during writing.

Supported formats: .png .jpg .jpeg .webp .gif .bmp
Vision models required: llava:7b, llama3.2-vision, moondream, etc.
Fallback (no vision model): stores a minimal metadata record only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from norvel_writer.config.defaults import IMAGE_FORMATS
from norvel_writer.ingestion.base import BaseIngestor, IngestedDocument

log = logging.getLogger(__name__)

# Prompt variants keyed by doc_type — tells the model what to focus on
_PROMPTS: dict[str, str] = {
    "visual": (
        "Describe this image in thorough detail. "
        "Include: all named locations, terrain features, bodies of water, roads, borders, "
        "settlements, points of interest, compass directions, scale hints, and any text or "
        "labels visible. Write as a reference entry a novelist can use."
    ),
    "codex": (
        "This image is worldbuilding reference material. Describe every visible detail: "
        "architecture, symbols, artefacts, geography, inscriptions, and atmosphere. "
        "Write as a codex entry."
    ),
    "character": (
        "Describe this character's appearance in detail: face, hair, eyes, build, "
        "clothing, accessories, weapons or tools, expression, posture, and any "
        "distinguishing marks. Write as a character reference."
    ),
    "default": (
        "Describe this image in detail. Focus on everything useful for a novelist: "
        "locations, characters, objects, spatial relationships, atmosphere, colours, "
        "and any visible text. Be thorough and specific."
    ),
}


class ImageIngestor(BaseIngestor):
    """
    Ingestor for image files.

    Strategy:
    1. Try to use a configured vision model to generate a rich text description.
    2. If no vision model is configured or available, store a placeholder
       with basic file metadata so the image is at least tracked.

    The generated description is what gets chunked, embedded, and stored — not
    the raw pixels. This makes image knowledge retrievable by semantic search
    exactly like any other project document.
    """

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in IMAGE_FORMATS

    def ingest(self, path: Path) -> IngestedDocument:
        # Synchronous entry point — actual vision call happens in the async pipeline.
        # Return a placeholder; the pipeline will call ingest_async if available.
        return IngestedDocument(
            text=f"[Image: {path.name}] (description pending — vision model required)",
            title=path.stem,
            metadata={"image_path": str(path), "needs_vision": "true"},
        )

    async def ingest_async(
        self,
        path: Path,
        vision_model: str,
        doc_type: str = "visual",
        language: str = "English",
    ) -> IngestedDocument:
        """
        Async variant used by IngestPipeline when a vision model is configured.
        Returns an IngestedDocument whose text is the model's description.
        """
        from norvel_writer.llm.ollama_client import get_client, OllamaModelNotFoundError

        base_prompt = _PROMPTS.get(doc_type, _PROMPTS["default"])
        # Language instruction placed FIRST so the model prioritises it
        # before reading the task description — prevents defaulting to English.
        if language.lower() in ("english", "en"):
            lang_prefix = "Write your response in English. "
        else:
            lang_prefix = (
                f"Write your ENTIRE response in {language}. "
                f"Do NOT respond in English. "
            )
        prompt = lang_prefix + base_prompt

        client = get_client()
        try:
            description = await client.describe_image(
                image_path=path,
                model=vision_model,
                prompt=prompt,
                language=language,
            )
            if not description.strip():
                description = f"[No description returned for {path.name}]"
        except OllamaModelNotFoundError as exc:
            log.warning("Vision model not available: %s", exc)
            description = (
                f"[Image: {path.name}]\n"
                f"Vision model '{vision_model}' is not installed. "
                f"Pull it with: ollama pull {vision_model}\n"
                f"File path: {path}"
            )
        except Exception as exc:
            log.error("Vision description failed for %s: %s", path.name, exc)
            description = f"[Image: {path.name}]\nDescription failed: {exc}"

        return IngestedDocument(
            text=description,
            title=path.stem,
            metadata={
                "image_path": str(path),
                "vision_model": vision_model,
                "doc_type": doc_type,
            },
        )
