"""VersionRepo: CRUD for chapter rewrite versions stored in SQLite."""
from __future__ import annotations

from typing import Dict, List, Optional


class VersionRepo:
    def __init__(self, db) -> None:
        self._db = db

    # ── Read ──────────────────────────────────────────────────────────────────

    def list_versions(self, chapter_id: str) -> List[Dict]:
        rows = self._db.execute(
            "SELECT * FROM chapter_versions "
            "WHERE chapter_id=? ORDER BY sort_order ASC, created_at ASC",
            (chapter_id,),
        )
        return [self._to_dict(r) for r in rows]

    def get_version(self, version_id: str) -> Optional[Dict]:
        row = self._db.execute_one(
            "SELECT * FROM chapter_versions WHERE id=?", (version_id,)
        )
        return self._to_dict(row) if row else None

    # ── Write ─────────────────────────────────────────────────────────────────

    def create_version(
        self,
        chapter_id: str,
        id: str,
        label: str,
        content: str,
        is_sheet: bool,
        sort_order: int,
        created_at: str,
    ) -> Dict:
        self._db.execute(
            """INSERT INTO chapter_versions
               (id, chapter_id, label, content, is_sheet, sort_order, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (id, chapter_id, label, content, int(is_sheet), sort_order, created_at),
        )
        return self.get_version(id) or {}

    def update_version(self, version_id: str, **kwargs) -> None:
        allowed = {"label", "content", "is_sheet", "sort_order"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "is_sheet" in updates:
            updates["is_sheet"] = int(updates["is_sheet"])
        sets = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [version_id]
        self._db.execute(
            f"UPDATE chapter_versions SET {sets} WHERE id=?", tuple(vals)
        )

    def delete_version(self, version_id: str) -> None:
        self._db.execute(
            "DELETE FROM chapter_versions WHERE id=?", (version_id,)
        )

    def update_labels(self, label_map: Dict[str, str]) -> None:
        """Update labels for multiple versions. label_map: {version_id: new_label}"""
        for vid, label in label_map.items():
            self._db.execute(
                "UPDATE chapter_versions SET label=? WHERE id=?", (label, vid)
            )

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(row) -> Dict:
        d = dict(row)
        d["is_sheet"] = bool(d.get("is_sheet", 0))
        return d
