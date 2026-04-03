"""SQLite schema migrations. Each entry is (version, sql_statements)."""
from __future__ import annotations

from typing import List, Tuple

# Each migration runs when schema_version < version.
# SQL may contain multiple statements separated by semicolons.
MIGRATIONS: List[Tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS projects (
            id                TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            description       TEXT,
            language          TEXT NOT NULL DEFAULT 'en',
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL,
            style_profile_id  TEXT
        );

        CREATE TABLE IF NOT EXISTS chapters (
            id          TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            title       TEXT NOT NULL,
            position    INTEGER NOT NULL,
            summary     TEXT,
            word_count  INTEGER DEFAULT 0,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            chapter_id  TEXT REFERENCES chapters(id),
            file_path   TEXT NOT NULL,
            file_hash   TEXT NOT NULL,
            doc_type    TEXT NOT NULL,
            format      TEXT NOT NULL,
            title       TEXT,
            language    TEXT,
            chunk_count INTEGER DEFAULT 0,
            ingested_at TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending','processing','ready','error'))
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            position    INTEGER NOT NULL,
            text        TEXT NOT NULL,
            token_count INTEGER
        );

        CREATE TABLE IF NOT EXISTS style_profiles (
            id           TEXT PRIMARY KEY,
            project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            name         TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            model_used   TEXT NOT NULL,
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS drafts (
            id          TEXT PRIMARY KEY,
            chapter_id  TEXT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
            content     TEXT NOT NULL,
            prompt_used TEXT,
            model_used  TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            is_accepted INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS external_edits (
            id            TEXT PRIMARY KEY,
            chapter_id    TEXT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
            export_path   TEXT NOT NULL,
            export_hash   TEXT NOT NULL,
            export_format TEXT NOT NULL,
            exported_at   TEXT NOT NULL,
            imported_at   TEXT,
            import_hash   TEXT,
            diff_json     TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document   ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_chapters_project  ON chapters(project_id);
        CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
        CREATE INDEX IF NOT EXISTS idx_drafts_chapter    ON drafts(chapter_id);
        """,
    ),
    (
        2,
        """
        ALTER TABLE projects ADD COLUMN persona TEXT NOT NULL DEFAULT '';
        """,
    ),
    (
        3,
        """
        ALTER TABLE chapters ADD COLUMN beats TEXT NOT NULL DEFAULT '';
        """,
    ),
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS chapter_images (
            id             TEXT PRIMARY KEY,
            chapter_id     TEXT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
            filename       TEXT NOT NULL,
            title          TEXT NOT NULL DEFAULT '',
            ai_description TEXT NOT NULL DEFAULT '',
            file_path      TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chapter_images_chapter ON chapter_images(chapter_id);
        """,
    ),
    (
        5,
        """
        CREATE TABLE IF NOT EXISTS project_images (
            id             TEXT PRIMARY KEY,
            project_id     TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            filename       TEXT NOT NULL,
            title          TEXT NOT NULL DEFAULT '',
            ai_description TEXT NOT NULL DEFAULT '',
            file_path      TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_project_images_project ON project_images(project_id);
        """,
    ),
]
