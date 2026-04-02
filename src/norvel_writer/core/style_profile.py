"""StyleProfileEngine: extract and store author style profiles."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class StyleProfile:
    avg_sentence_length: str = ""
    vocabulary_richness: str = ""
    narrative_distance: str = ""          # close 3rd, distant 3rd, 1st, etc.
    tense: str = ""
    pacing: str = ""
    imagery_density: str = ""
    dialogue_habits: str = ""
    paragraph_rhythm: str = ""
    tone_markers: List[str] = field(default_factory=list)
    structural_preferences: str = ""
    example_phrases: List[str] = field(default_factory=list)
    avoid_patterns: List[str] = field(default_factory=list)
    raw_notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "StyleProfile":
        d = json.loads(data)
        return cls(**d)

    def to_prompt_summary(self) -> str:
        lines = []
        if self.avg_sentence_length:
            lines.append(f"Sentence length: {self.avg_sentence_length}")
        if self.narrative_distance:
            lines.append(f"Narrative distance: {self.narrative_distance}")
        if self.tense:
            lines.append(f"Tense: {self.tense}")
        if self.pacing:
            lines.append(f"Pacing: {self.pacing}")
        if self.tone_markers:
            lines.append(f"Tone: {', '.join(self.tone_markers)}")
        if self.dialogue_habits:
            lines.append(f"Dialogue: {self.dialogue_habits}")
        if self.imagery_density:
            lines.append(f"Imagery: {self.imagery_density}")
        if self.paragraph_rhythm:
            lines.append(f"Paragraph rhythm: {self.paragraph_rhythm}")
        if self.example_phrases:
            lines.append("Example phrases: " + "; ".join(self.example_phrases[:3]))
        if self.avoid_patterns:
            lines.append("Avoid: " + "; ".join(self.avoid_patterns[:3]))
        return "\n".join(lines)


class StyleProfileEngine:
    """
    Build and update style profiles from author sample documents.
    """

    def __init__(self, db=None, model: Optional[str] = None) -> None:
        from norvel_writer.storage.db import get_db
        from norvel_writer.config.settings import get_config
        self._db = db or get_db()
        self._model = model or get_config().default_chat_model

    async def build_profile(
        self,
        project_id: str,
        profile_name: str = "Default Style",
        progress_cb=None,
    ) -> str:
        """
        Build a style profile from all style_sample documents in the project.
        Returns the new style profile ID.
        """
        from norvel_writer.storage.repositories.document_repo import DocumentRepo
        from norvel_writer.storage.repositories.style_repo import StyleRepo
        from norvel_writer.llm.langchain_bridge import chat_complete
        from norvel_writer.llm.prompt_builder import build_style_extraction_messages

        doc_repo = DocumentRepo(self._db)
        style_repo = StyleRepo(self._db)

        docs = doc_repo.list_documents(project_id, doc_type="style_sample")
        if not docs:
            log.warning("No style samples found for project %s", project_id)
            # Return empty profile
            profile = StyleProfile(raw_notes="No style samples provided.")
        else:
            # Gather sample texts (first chunk of each doc)
            sample_texts: List[str] = []
            for doc in docs[:20]:  # cap at 20 samples
                chunks = doc_repo.list_chunks(doc["id"])
                if chunks:
                    sample_texts.append(chunks[0]["text"])

            if progress_cb:
                progress_cb(20)

            # Process in batches of 5
            all_notes: List[str] = []
            batch_size = 5
            for i in range(0, len(sample_texts), batch_size):
                batch = sample_texts[i : i + batch_size]
                messages = build_style_extraction_messages(batch)
                try:
                    response = await chat_complete(messages)
                    all_notes.append(response)
                except Exception as exc:
                    log.error("Style extraction failed for batch %d: %s", i, exc)

                if progress_cb:
                    done = min(i + batch_size, len(sample_texts))
                    pct = 20 + int((done / len(sample_texts)) * 60)
                    progress_cb(pct)

            # Synthesise notes into structured profile
            profile = await self._synthesise_profile(all_notes)

        if progress_cb:
            progress_cb(90)

        profile_id = style_repo.create_style_profile(
            project_id=project_id,
            name=profile_name,
            profile_json=profile.to_json(),
            model_used=self._model,
        )

        if progress_cb:
            progress_cb(100)

        log.info("Built style profile %s for project %s", profile_id, project_id)
        return profile_id

    async def _synthesise_profile(
        self, notes: List[str]
    ) -> StyleProfile:
        """Merge multiple style analysis notes into one structured profile."""
        if not notes:
            return StyleProfile()

        combined = "\n\n---\n\n".join(notes)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a literary style analyst. Based on the following style analysis notes, "
                    "produce a JSON object with these exact keys:\n"
                    "avg_sentence_length, vocabulary_richness, narrative_distance, tense, pacing, "
                    "imagery_density, dialogue_habits, paragraph_rhythm, tone_markers (array), "
                    "structural_preferences, example_phrases (array of up to 5), "
                    "avoid_patterns (array), raw_notes.\n"
                    "Return ONLY valid JSON, no markdown."
                ),
            },
            {"role": "user", "content": combined},
        ]
        try:
            from norvel_writer.llm.langchain_bridge import chat_complete
            response = await chat_complete(messages)
            # Strip any markdown code fences
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text.strip())
            return StyleProfile(**{k: data.get(k, v) for k, v in asdict(StyleProfile()).items()})
        except Exception as exc:
            log.error("Failed to synthesise style profile: %s", exc)
            return StyleProfile(raw_notes=combined[:500])
