"""ExternalEditRoundTrip: export, watch, import, diff."""
from __future__ import annotations

import difflib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

from norvel_writer.utils.text_utils import hash_text

log = logging.getLogger(__name__)


@dataclass
class DiffChunk:
    tag: str          # 'equal', 'insert', 'delete', 'replace'
    original: str
    revised: str


class DiffEngine:
    """
    Manages the external edit round-trip:
    export → watch → import → diff → continue from revised.
    """

    def __init__(self, db=None) -> None:
        from norvel_writer.storage.db import get_db
        self._db = db or get_db()

    def export_for_editing(
        self,
        chapter_id: str,
        content: str,
        fmt: str = "md",
    ) -> Path:
        """Write content to a temp file for external editing. Returns path."""
        from norvel_writer.config.settings import get_config
        from norvel_writer.storage.repositories.draft_repo import DraftRepo

        cfg = get_config()
        # Get project_id for this chapter
        from norvel_writer.storage.repositories.project_repo import ProjectRepo
        pr = ProjectRepo(self._db)
        chapter = pr.get_chapter(chapter_id)
        if not chapter:
            raise ValueError(f"Chapter {chapter_id} not found")

        project_id = chapter["project_id"]
        edit_dir = cfg.projects_path / project_id / "external_edits"
        edit_dir.mkdir(parents=True, exist_ok=True)

        ext = ".md" if fmt == "md" else ".txt"
        out_path = edit_dir / f"chapter_{chapter_id[:8]}_edit{ext}"
        out_path.write_text(content, encoding="utf-8")

        # Record the export
        dr = DraftRepo(self._db)
        dr.create_external_edit(
            chapter_id=chapter_id,
            export_path=str(out_path),
            export_hash=hash_text(content),
            export_format=fmt,
        )

        log.info("Exported chapter %s to %s", chapter_id, out_path)
        return out_path

    def import_edited(
        self,
        chapter_id: str,
        file_path: Optional[Path] = None,
        content: Optional[str] = None,
    ) -> List[DiffChunk]:
        """
        Import externally edited content. Returns diff against original export.
        Pass either file_path (reads file) or content (raw text).
        """
        from norvel_writer.storage.repositories.draft_repo import DraftRepo

        if file_path:
            content = file_path.read_text(encoding="utf-8")
        if content is None:
            raise ValueError("Must provide file_path or content")

        dr = DraftRepo(self._db)
        edit_record = dr.get_latest_edit(chapter_id)
        if not edit_record:
            # No prior export — treat as fresh draft
            return []

        original_hash = edit_record["export_hash"]
        import_hash = hash_text(content)

        # Compute diff
        original_content = self._recover_original(edit_record)
        diff_chunks = compute_diff(original_content or "", content)

        dr.record_import(
            edit_id=edit_record["id"],
            import_hash=import_hash,
            diff_json=json.dumps([asdict(c) for c in diff_chunks]),
        )

        log.info(
            "Imported edit for chapter %s (%d diff chunks)", chapter_id, len(diff_chunks)
        )
        return diff_chunks

    def _recover_original(self, edit_record: dict) -> Optional[str]:
        """Try to read the original exported file."""
        path = Path(edit_record["export_path"])
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def watch_file(
        self,
        path: Path,
        on_change: Callable[[Path], None],
    ) -> object:
        """
        Start watching a file for changes. Returns a watchdog observer.
        Caller is responsible for stopping it.
        """
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and Path(event.src_path) == path:
                    on_change(path)

        observer = Observer()
        observer.schedule(_Handler(), str(path.parent), recursive=False)
        observer.start()
        return observer


def compute_diff(original: str, revised: str) -> List[DiffChunk]:
    """Compute character-level diff between two texts."""
    matcher = difflib.SequenceMatcher(None, original, revised, autojunk=False)
    chunks: List[DiffChunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunks.append(DiffChunk(
            tag=tag,
            original=original[i1:i2],
            revised=revised[j1:j2],
        ))
    return chunks
