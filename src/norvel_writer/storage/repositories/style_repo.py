"""CRUD for style profiles."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from norvel_writer.storage.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StyleRepo:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create_style_profile(
        self,
        project_id: str,
        name: str,
        profile_json: str,
        model_used: str,
    ) -> str:
        sid = str(uuid.uuid4())
        now = _now()
        with self._db.connect() as conn:
            conn.execute(
                """INSERT INTO style_profiles(id, project_id, name, profile_json, model_used, created_at)
                   VALUES(?,?,?,?,?,?)""",
                (sid, project_id, name, profile_json, model_used, now),
            )
        return sid

    def get_style_profile(self, profile_id: str) -> Optional[dict]:
        row = self._db.execute_one(
            "SELECT * FROM style_profiles WHERE id=?", (profile_id,)
        )
        return dict(row) if row else None

    def list_style_profiles(self, project_id: str) -> List[dict]:
        rows = self._db.execute(
            "SELECT * FROM style_profiles WHERE project_id=? ORDER BY created_at DESC",
            (project_id,),
        )
        return [dict(r) for r in rows]

    def update_style_profile(self, profile_id: str, profile_json: str) -> None:
        with self._db.connect() as conn:
            conn.execute(
                "UPDATE style_profiles SET profile_json=? WHERE id=?",
                (profile_json, profile_id),
            )

    def delete_style_profile(self, profile_id: str) -> None:
        with self._db.connect() as conn:
            conn.execute("DELETE FROM style_profiles WHERE id=?", (profile_id,))

    def get_active_profile(self, project_id: str) -> Optional[dict]:
        """Return the profile linked to the project, or the most recent one."""
        row = self._db.execute_one(
            """SELECT sp.* FROM style_profiles sp
               JOIN projects p ON p.style_profile_id = sp.id
               WHERE p.id=?""",
            (project_id,),
        )
        if row:
            return dict(row)
        # Fallback: most recent
        row = self._db.execute_one(
            "SELECT * FROM style_profiles WHERE project_id=? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        return dict(row) if row else None
