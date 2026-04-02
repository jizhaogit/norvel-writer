"""CRUD for projects and chapters."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from norvel_writer.storage.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ── Projects ──────────────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        description: str = "",
        language: str = "en",
    ) -> str:
        pid = str(uuid.uuid4())
        now = _now()
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO projects(id, name, description, language, created_at, updated_at)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (pid, name, description, language, now, now),
            )
        return pid

    def get_project(self, project_id: str) -> Optional[dict]:
        row = self._db.execute_one(
            "SELECT * FROM projects WHERE id=?", (project_id,)
        )
        return dict(row) if row else None

    def list_projects(self) -> List[dict]:
        rows = self._db.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC"
        )
        return [dict(r) for r in rows]

    def update_project(self, project_id: str, **kwargs) -> None:
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [project_id]
        with self._db.connect() as conn:
            conn.execute(f"UPDATE projects SET {sets} WHERE id=?", vals)

    def delete_project(self, project_id: str) -> None:
        with self._db.connect() as conn:
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))

    # ── Chapters ──────────────────────────────────────────────────────────

    def create_chapter(
        self,
        project_id: str,
        title: str,
        position: Optional[int] = None,
    ) -> str:
        cid = str(uuid.uuid4())
        now = _now()
        if position is None:
            row = self._db.execute_one(
                "SELECT COALESCE(MAX(position),0)+1 AS pos FROM chapters WHERE project_id=?",
                (project_id,),
            )
            position = row["pos"] if row else 1
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO chapters(id, project_id, title, position, created_at, updated_at)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (cid, project_id, title, position, now, now),
            )
        return cid

    def get_chapter(self, chapter_id: str) -> Optional[dict]:
        row = self._db.execute_one(
            "SELECT * FROM chapters WHERE id=?", (chapter_id,)
        )
        return dict(row) if row else None

    def list_chapters(self, project_id: str) -> List[dict]:
        rows = self._db.execute(
            "SELECT * FROM chapters WHERE project_id=? ORDER BY position",
            (project_id,),
        )
        return [dict(r) for r in rows]

    def update_chapter(self, chapter_id: str, **kwargs) -> None:
        kwargs["updated_at"] = _now()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [chapter_id]
        with self._db.connect() as conn:
            conn.execute(f"UPDATE chapters SET {sets} WHERE id=?", vals)

    def delete_chapter(self, chapter_id: str) -> None:
        with self._db.connect() as conn:
            conn.execute("DELETE FROM chapters WHERE id=?", (chapter_id,))
