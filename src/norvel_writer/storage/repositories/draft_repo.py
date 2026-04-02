"""CRUD for drafts and external edits."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from norvel_writer.storage.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DraftRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create_draft(
        self,
        chapter_id: str,
        content: str,
        model_used: str,
        prompt_used: Optional[str] = None,
    ) -> str:
        did = str(uuid.uuid4())
        now = _now()
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO drafts(id, chapter_id, content, prompt_used, model_used, created_at)
                   VALUES(?,?,?,?,?,?)""",
                (did, chapter_id, content, prompt_used, model_used, now),
            )
        return did

    def get_draft(self, draft_id: str) -> Optional[dict]:
        row = self._db.execute_one("SELECT * FROM drafts WHERE id=?", (draft_id,))
        return dict(row) if row else None

    def list_drafts(self, chapter_id: str) -> List[dict]:
        rows = self._db.execute(
            "SELECT * FROM drafts WHERE chapter_id=? ORDER BY created_at DESC",
            (chapter_id,),
        )
        return [dict(r) for r in rows]

    def accept_draft(self, draft_id: str) -> None:
        with self._db.connect() as conn:
            # Unaccept all siblings first
            row = conn.execute(
                "SELECT chapter_id FROM drafts WHERE id=?", (draft_id,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE drafts SET is_accepted=0 WHERE chapter_id=?",
                    (row["chapter_id"],),
                )
            conn.execute(
                "UPDATE drafts SET is_accepted=1 WHERE id=?", (draft_id,)
            )

    def get_accepted_draft(self, chapter_id: str) -> Optional[dict]:
        row = self._db.execute_one(
            "SELECT * FROM drafts WHERE chapter_id=? AND is_accepted=1",
            (chapter_id,),
        )
        return dict(row) if row else None

    def delete_draft(self, draft_id: str) -> None:
        with self._db.connect() as conn:
            conn.execute("DELETE FROM drafts WHERE id=?", (draft_id,))

    # ── External Edits ────────────────────────────────────────────────────

    def create_external_edit(
        self,
        chapter_id: str,
        export_path: str,
        export_hash: str,
        export_format: str,
    ) -> str:
        eid = str(uuid.uuid4())
        now = _now()
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO external_edits
                   (id, chapter_id, export_path, export_hash, export_format, exported_at)
                   VALUES(?,?,?,?,?,?)""",
                (eid, chapter_id, export_path, export_hash, export_format, now),
            )
        return eid

    def record_import(
        self,
        edit_id: str,
        import_hash: str,
        diff_json: Optional[str] = None,
    ) -> None:
        now = _now()
        with self._db.connect() as conn:
            conn.execute(
                """UPDATE external_edits
                   SET imported_at=?, import_hash=?, diff_json=?
                   WHERE id=?""",
                (now, import_hash, diff_json, edit_id),
            )

    def get_latest_edit(self, chapter_id: str) -> Optional[dict]:
        row = self._db.execute_one(
            "SELECT * FROM external_edits WHERE chapter_id=? ORDER BY exported_at DESC LIMIT 1",
            (chapter_id,),
        )
        return dict(row) if row else None
