"""FastAPI application for Norvel Writer."""
from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from html.parser import HTMLParser
from pydantic import BaseModel

log = logging.getLogger(__name__)

app = FastAPI(title="Norvel Writer", version="0.1.0")


# ── HTML → plain-text helper ───────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text converter that preserves paragraph breaks."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._block_tags = {
            "p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6",
            "tr", "blockquote",
        }

    def handle_starttag(self, tag, attrs):
        if tag in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data):
        self._parts.append(data)

    def handle_entityref(self, name):
        import html
        self._parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        import html
        self._parts.append(html.unescape(f"&#{name};"))

    def get_text(self) -> str:
        import re
        text = "".join(self._parts)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _strip_html(html_content: str) -> str:
    """Strip HTML tags and return clean plain text."""
    if not html_content:
        return ""
    # Fast path: if there are no tags at all, return as-is
    if "<" not in html_content:
        return html_content.strip()
    stripper = _HTMLStripper()
    stripper.feed(html_content)
    return stripper.get_text()

# ── Singleton ProjectManager ───────────────────────────────────────────────

_pm: Optional[Any] = None


def get_pm():
    global _pm
    if _pm is None:
        from norvel_writer.core.project import ProjectManager
        _pm = ProjectManager()
    return _pm


# ── Helpers ────────────────────────────────────────────────────────────────

def _web_dir() -> Path:
    return Path(__file__).parent.parent / "web"


async def _sse_stream(gen: AsyncIterator[str]) -> AsyncIterator[str]:
    try:
        async for chunk in gen:
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as exc:
        log.error("SSE stream error: %s", exc)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


# ── Static ─────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    html_path = _web_dir() / "index.html"
    return FileResponse(str(html_path), media_type="text/html")


# ── Ollama ─────────────────────────────────────────────────────────────────

@app.get("/api/ollama/status")
async def ollama_status():
    try:
        from norvel_writer.llm.model_manager import get_ollama_status
        status = await get_ollama_status()
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        vision_model = cfg.vision_model.strip()
        model_names = [m.name for m in status.models]
        vision_available = bool(vision_model) and any(
            n == vision_model or n.startswith(vision_model.split(":")[0])
            for n in model_names
        )
        return {
            "installed": status.installed,
            "running": status.running,
            "version": status.version,
            "models": [{"name": m.name, "size": getattr(m, "size", 0)} for m in status.models],
            "vision_model": vision_model,
            "vision_available": vision_available,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model/health")
async def model_health():
    """
    Check whether the configured chat and embed models are actually available.
    For Ollama providers: verifies the model name exists in the local model list.
    For cloud providers: verifies that the relevant API key is set.
    Returns {chat: {ok, name, error}, embed: {ok, name, error}}.
    """
    from dotenv import dotenv_values
    from norvel_writer.llm.langchain_bridge import find_env_path

    env_path = find_env_path()
    env = dotenv_values(env_path) if (env_path and env_path.exists()) else {}

    provider       = env.get("LLM_PROVIDER", "ollama").strip().lower()
    embed_provider = env.get("EMBEDDINGS_PROVIDER", provider).strip().lower()

    chat_r:  dict = {"ok": False, "name": "", "error": ""}
    embed_r: dict = {"ok": False, "name": "", "error": ""}

    # ── helper: check Ollama model list ──────────────────────────────────────
    _ollama_models: list[str] | None = None

    async def _ollama_model_names() -> list[str]:
        nonlocal _ollama_models
        if _ollama_models is None:
            try:
                from norvel_writer.llm.ollama_client import get_client
                raw = await get_client().list_models()
                _ollama_models = [getattr(m, "model", getattr(m, "name", str(m))) for m in raw]
            except Exception:
                _ollama_models = []
        return _ollama_models

    def _model_in_list(name: str, names: list[str]) -> bool:
        if not name:
            return False
        if name in names:
            return True
        base = name.split(":")[0]
        return any(n == name or n.startswith(base + ":") for n in names)

    # ── Chat model ────────────────────────────────────────────────────────────
    if provider == "ollama":
        name = env.get("OLLAMA_CHAT_MODEL", "").strip()
        chat_r["name"] = name
        names = await _ollama_model_names()
        if _model_in_list(name, names):
            chat_r["ok"] = True
        else:
            chat_r["error"] = f"Not pulled — run: ollama pull {name}" if name else "No model configured"
    elif provider == "openai":
        key  = env.get("OPENAI_API_KEY", "").strip()
        chat_r["name"] = env.get("OPENAI_CHAT_MODEL", "").strip()
        chat_r["ok"]   = bool(key)
        if not key:
            chat_r["error"] = "OPENAI_API_KEY not set in .env"
    elif provider == "anthropic":
        key  = env.get("ANTHROPIC_API_KEY", "").strip()
        chat_r["name"] = env.get("ANTHROPIC_MODEL", "").strip()
        chat_r["ok"]   = bool(key)
        if not key:
            chat_r["error"] = "ANTHROPIC_API_KEY not set in .env"
    elif provider == "gemini":
        key  = env.get("GOOGLE_API_KEY", "").strip()
        chat_r["name"] = env.get("GEMINI_MODEL", "").strip()
        chat_r["ok"]   = bool(key)
        if not key:
            chat_r["error"] = "GOOGLE_API_KEY not set in .env"
    else:
        chat_r["error"] = f"Unknown provider: {provider}"

    # ── Embed model ───────────────────────────────────────────────────────────
    if embed_provider == "ollama":
        name = env.get("OLLAMA_EMBED_MODEL", "").strip()
        embed_r["name"] = name
        names = await _ollama_model_names()
        if _model_in_list(name, names):
            embed_r["ok"] = True
        else:
            embed_r["error"] = f"Not pulled — run: ollama pull {name}" if name else "No model configured"
    elif embed_provider == "openai":
        key  = env.get("OPENAI_API_KEY", "").strip()
        embed_r["name"] = env.get("OPENAI_EMBED_MODEL", "").strip()
        embed_r["ok"]   = bool(key)
        if not key:
            embed_r["error"] = "OPENAI_API_KEY not set in .env"
    else:
        embed_r["error"] = f"Unknown embed provider: {embed_provider}"

    return {"chat": chat_r, "embed": embed_r}


@app.post("/api/ollama/start")
async def ollama_start():
    try:
        from norvel_writer.llm.model_manager import start_ollama_serve
        ok = await start_ollama_serve()
        return {"ok": ok}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ollama/pull/{model:path}")
async def ollama_pull(model: str):
    """Stream model pull progress as SSE.

    Uses httpx directly against the configured Ollama REST API so that:
    - The correct host/port (from .env OLLAMA_BASE_URL / OLLAMA_HOST) is used
    - No SDK-level timeout kills a slow download
    - A keepalive comment is sent every ~5 seconds so the browser SSE
      connection stays alive during long pauses between progress events
    """
    async def _gen():
        import asyncio
        import httpx
        import os

        # Resolve Ollama base URL: OLLAMA_HOST (legacy) → OLLAMA_BASE_URL → default
        base_url = (
            os.environ.get("OLLAMA_HOST", "").strip()
            or os.environ.get("OLLAMA_BASE_URL", "").strip()
            or "http://127.0.0.1:11434"
        )
        if "://" not in base_url:
            base_url = "http://" + base_url
        base_url = base_url.rstrip("/")

        try:
            # timeout=None — large models can take many minutes to download
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/api/pull",
                    json={"model": model, "stream": True},
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield f"data: {json.dumps({'error': f'Ollama returned {resp.status_code}: {body.decode()[:200]}'})}\n\n"
                        return

                    last_event = asyncio.get_event_loop().time()
                    async for line in resp.aiter_lines():
                        now = asyncio.get_event_loop().time()
                        # Send SSE keepalive comment if silent for >4 s
                        if now - last_event > 4:
                            yield ": keepalive\n\n"
                        last_event = now

                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        status = data.get("status", "")
                        completed = data.get("completed")
                        total = data.get("total")
                        pct = 0
                        if completed and total and total > 0:
                            pct = int(completed / total * 100)
                        yield f"data: {json.dumps({'status': status, 'pct': pct})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ── Projects ───────────────────────────────────────────────────────────────

@app.get("/api/projects")
async def list_projects():
    try:
        return get_pm().list_projects()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    language: str = "en"


@app.post("/api/projects")
async def create_project(body: ProjectCreate):
    try:
        pid = get_pm().create_project(body.name, body.description, body.language)
        return {"id": pid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    try:
        proj = get_pm().get_project(project_id)
        if proj is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return proj
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, body: ProjectUpdate):
    try:
        updates = {k: v for k, v in body.model_dump().items() if v is not None}
        if not updates:
            return {"ok": True}
        get_pm().update_project(project_id, **updates)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    try:
        get_pm().delete_project(project_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Chapters ───────────────────────────────────────────────────────────────

@app.get("/api/projects/{project_id}/chapters")
async def list_chapters(project_id: str):
    try:
        return get_pm().list_chapters(project_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ChapterCreate(BaseModel):
    title: str


@app.post("/api/projects/{project_id}/chapters")
async def create_chapter(project_id: str, body: ChapterCreate):
    try:
        cid = get_pm().create_chapter(project_id, body.title)
        return {"id": cid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/chapters/{chapter_id}")
async def get_chapter(chapter_id: str):
    try:
        ch = get_pm().get_chapter(chapter_id)
        if ch is None:
            raise HTTPException(status_code=404, detail="Chapter not found")
        return ch
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/chapters/{chapter_id}")
async def update_chapter(chapter_id: str, body: Dict[str, Any]):
    try:
        get_pm().update_chapter(chapter_id, **body)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/chapters/{chapter_id}")
async def delete_chapter(chapter_id: str):
    try:
        get_pm().delete_chapter(chapter_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/chapters/{chapter_id}/content")
async def get_chapter_content(chapter_id: str):
    try:
        draft = get_pm().get_accepted_draft(chapter_id)
        content = draft["content"] if draft else ""
        return {"content": content}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ContentUpdate(BaseModel):
    content: str
    model_used: str = "manual"


@app.put("/api/chapters/{chapter_id}/content")
async def update_chapter_content(chapter_id: str, body: ContentUpdate):
    try:
        pm = get_pm()
        draft_id = pm.save_draft(chapter_id, body.content, body.model_used)
        pm.accept_draft(draft_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Documents ──────────────────────────────────────────────────────────────

@app.get("/api/projects/{project_id}/documents")
async def list_documents(
    project_id: str,
    doc_type: Optional[str] = Query(default=None),
):
    try:
        return get_pm().list_documents(project_id, doc_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects/{project_id}/ingest")
async def ingest_document(
    project_id: str,
    file: UploadFile,
    doc_type: str = Form("notes"),
):
    from norvel_writer.config.settings import get_config
    cfg = get_config()

    # Permanently store the uploaded file in the project's files directory
    files_dir = cfg.projects_path / project_id / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "upload").name
    dest_path = files_dir / safe_name
    # Avoid overwriting an existing file with the same name
    if dest_path.exists():
        stem = dest_path.stem
        suffix = dest_path.suffix
        i = 1
        while dest_path.exists():
            dest_path = files_dir / f"{stem}_{i}{suffix}"
            i += 1

    try:
        content = await file.read()
        dest_path.write_bytes(content)

        from norvel_writer.ingestion.pipeline import IngestPipeline
        pipeline = IngestPipeline()
        doc_id = await pipeline.run(
            file_path=dest_path,
            project_id=project_id,
            doc_type=doc_type,
        )
        return {"id": doc_id, "ok": True, "stored_path": str(dest_path)}
    except Exception as exc:
        # Clean up the saved file if ingestion failed
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, project_id: str = Query(...)):
    try:
        get_pm().delete_document(doc_id, project_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/documents/{doc_id}/content")
async def get_document_content(doc_id: str):
    """Return the full text of a document by joining its stored chunks."""
    try:
        from norvel_writer.storage.repositories.document_repo import DocumentRepo
        from norvel_writer.storage.db import get_db
        repo = DocumentRepo(get_db())
        chunks = repo.list_chunks(doc_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="Document has no content")
        text = "\n\n".join(c["text"] for c in chunks)
        doc = repo.get_document(doc_id)
        return {"text": text, "title": doc["title"] if doc else "", "doc_type": doc["doc_type"] if doc else ""}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class DocumentContentUpdate(BaseModel):
    text: str
    project_id: str


@app.put("/api/documents/{doc_id}/content")
async def update_document_content(doc_id: str, body: DocumentContentUpdate):
    """Re-chunk and re-embed edited document text, replacing the old content."""
    try:
        from norvel_writer.storage.repositories.document_repo import DocumentRepo
        from norvel_writer.storage.db import get_db
        from norvel_writer.storage.vector_store import get_vector_store
        from norvel_writer.llm.embedder import EmbeddingService
        from norvel_writer.utils.chunker import chunk_text

        db = get_db()
        vs = get_vector_store()
        repo = DocumentRepo(db)

        doc = repo.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        project_id = body.project_id
        doc_type = doc["doc_type"]

        # 1. Remove old chunks from vector store and SQLite
        vs.delete_by_document(f"project_{project_id}", doc_id)
        if doc_type == "style_sample":
            vs.delete_by_document(f"style_{project_id}", doc_id)
        repo.delete_chunks(doc_id)

        # 2. Re-chunk
        chunks = chunk_text(body.text, language="english")
        if not chunks:
            repo.update_document_status(doc_id, "ready", chunk_count=0)
            return {"ok": True, "chunks": 0}

        # 3. Insert new chunks into SQLite
        chunk_ids = repo.insert_chunks(doc_id, chunks)

        # 4. Re-embed
        embedder = EmbeddingService()
        embeddings = await embedder.embed_texts(chunks)

        # 5. Upsert into vector store
        metadatas = [
            {
                "document_id": doc_id,
                "doc_type": doc_type,
                "chapter_id": doc.get("chapter_id") or "",
                "position": str(i),
                "language": doc.get("language") or "",
                "title": doc.get("title") or "",
                "project_id": project_id,
                "is_image": "False",
            }
            for i in range(len(chunk_ids))
        ]
        vs.upsert_chunks(
            collection_name=f"project_{project_id}",
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        if doc_type == "style_sample":
            vs.upsert_chunks(
                collection_name=f"style_{project_id}",
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

        repo.update_document_status(doc_id, "ready", chunk_count=len(chunks))
        return {"ok": True, "chunks": len(chunks)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── AI: Continue / Rewrite / Chat ──────────────────────────────────────────

class ContinueRequest(BaseModel):
    chapter_id: str = ""
    current_text: str = ""          # text before the cursor
    text_after_cursor: str = ""     # text after the cursor (empty = cursor at end)
    user_instruction: str = "Continue the story from where it left off."
    style_mode: str = "inspired_by"
    language: str = "en"
    active_doc_types: Optional[List[str]] = None
    editor_note: str = ""           # pinned editor suggestion from the browser
    qa_note: str = ""               # pinned QA report from the browser


@app.post("/api/projects/{project_id}/continue")
async def continue_draft(project_id: str, body: ContinueRequest):
    async def _gen():
        try:
            from norvel_writer.core.draft_engine import DraftEngine
            engine = DraftEngine(project_manager=get_pm())
            # Load beats for this chapter from DB
            chapter_beats = ""
            if body.chapter_id:
                from norvel_writer.storage.repositories.project_repo import ProjectRepo
                from norvel_writer.storage.db import get_db
                ch_row = ProjectRepo(get_db()).get_chapter(body.chapter_id)
                chapter_beats = (ch_row.get("beats") or "").strip() if ch_row else ""
            stream = await engine.continue_draft(
                project_id=project_id,
                chapter_id=body.chapter_id,
                current_text=body.current_text,
                text_after_cursor=body.text_after_cursor,
                user_instruction=body.user_instruction,
                style_mode=body.style_mode,
                language=body.language,
                active_doc_types=body.active_doc_types,
                beats=chapter_beats,
                editor_note=body.editor_note,
                qa_note=body.qa_note,
            )
            async for chunk in stream:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except asyncio.CancelledError:
            # Client disconnected / request aborted — stop silently
            return
        except Exception as exc:
            log.error("continue_draft error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


class RewriteRequest(BaseModel):
    passage: str
    chapter_id: str = ""
    user_instruction: str = "Rewrite this passage in the same style."
    style_mode: str = "preserve_tone_rhythm"
    language: str = "en"
    editor_note: str = ""           # pinned editor suggestion from the browser
    qa_note: str = ""               # pinned QA report from the browser


@app.post("/api/projects/{project_id}/rewrite")
async def rewrite_passage(project_id: str, body: RewriteRequest):
    async def _gen():
        try:
            from norvel_writer.core.draft_engine import DraftEngine
            engine = DraftEngine(project_manager=get_pm())
            # Load beats for this chapter from DB (same as continue writing)
            chapter_beats = ""
            if body.chapter_id:
                from norvel_writer.storage.repositories.project_repo import ProjectRepo
                from norvel_writer.storage.db import get_db
                ch_row = ProjectRepo(get_db()).get_chapter(body.chapter_id)
                chapter_beats = (ch_row.get("beats") or "").strip() if ch_row else ""
            stream = await engine.rewrite_passage(
                project_id=project_id,
                passage=body.passage,
                chapter_id=body.chapter_id,
                beats=chapter_beats,
                user_instruction=body.user_instruction,
                style_mode=body.style_mode,
                language=body.language,
                editor_note=body.editor_note,
                qa_note=body.qa_note,
            )
            async for chunk in stream:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.error("rewrite_passage error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


class ChatRequest(BaseModel):
    question: str
    chapter_id: str = ""
    role: str = "editor"   # "editor" | "writer" | "qa"
    history: Optional[List[Dict[str, str]]] = None
    language: str = "en"
    editor_note: str = ""   # pinned editor suggestion forwarded from the browser
    qa_note: str = ""       # pinned QA report forwarded from the browser


@app.post("/api/projects/{project_id}/chat")
async def chat_with_context(project_id: str, body: ChatRequest):
    async def _gen():
        try:
            from norvel_writer.core.draft_engine import DraftEngine
            engine = DraftEngine(project_manager=get_pm())
            stream = await engine.chat_with_context(
                project_id=project_id,
                question=body.question,
                chapter_id=body.chapter_id,
                role=body.role,
                history=body.history,
                language=body.language,
                editor_note=body.editor_note,
                qa_note=body.qa_note,
            )
            async for chunk in stream:
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            log.error("chat_with_context error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ── Role file API ─────────────────────────────────────────────────────────

VALID_ROLES = {"editor", "writer", "qa"}

@app.get("/api/roles/{role}")
async def get_role_file(role: str):
    if role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Unknown role: {role}")
    from norvel_writer.core.role_loader import _user_roles_dir, _bundled_roles_dir, ensure_user_role_files
    ensure_user_role_files()
    user_path = _user_roles_dir() / f"{role}.toml"
    bundled_path = _bundled_roles_dir() / f"{role}.toml"
    path = user_path if user_path.exists() else bundled_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Role file not found")
    return {"role": role, "path": str(path), "content": path.read_text(encoding="utf-8")}

class RoleFileBody(BaseModel):
    content: str

@app.put("/api/roles/{role}")
async def save_role_file(role: str, body: RoleFileBody):
    if role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Unknown role: {role}")
    from norvel_writer.core.role_loader import _user_roles_dir
    user_dir = _user_roles_dir()
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{role}.toml"
    path.write_text(body.content, encoding="utf-8")
    return {"ok": True, "path": str(path)}

@app.delete("/api/roles/{role}")
async def reset_role_file(role: str):
    """Delete the user override so the bundled default is used again."""
    if role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Unknown role: {role}")
    from norvel_writer.core.role_loader import _user_roles_dir, _bundled_roles_dir
    user_path = _user_roles_dir() / f"{role}.toml"
    if user_path.exists():
        user_path.unlink()
    bundled_path = _bundled_roles_dir() / f"{role}.toml"
    content = bundled_path.read_text(encoding="utf-8") if bundled_path.exists() else ""
    return {"ok": True, "content": content}

# ── Chapter images ────────────────────────────────────────────────────────

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

def _images_dir(chapter_id: str) -> Path:
    from norvel_writer.config.settings import get_config
    d = get_config().chapter_images_path / chapter_id
    d.mkdir(parents=True, exist_ok=True)
    return d

@app.get("/api/chapters/{chapter_id}/images")
async def list_chapter_images(chapter_id: str):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM chapter_images WHERE chapter_id=? ORDER BY created_at ASC",
        (chapter_id,),
    )
    return [dict(r) for r in rows]

@app.post("/api/chapters/{chapter_id}/images")
async def upload_chapter_image(chapter_id: str, file: UploadFile, title: str = Form(default="")):
    import shutil, uuid
    from datetime import datetime, timezone

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")

    image_id = str(uuid.uuid4())
    filename  = f"{image_id}{ext}"
    dest      = _images_dir(chapter_id) / filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    now = datetime.now(timezone.utc).isoformat()
    display_title = title.strip() or file.filename or filename
    db = get_db()
    db.execute(
        "INSERT INTO chapter_images(id, chapter_id, filename, title, ai_description, file_path, created_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (image_id, chapter_id, filename, display_title, "", str(dest), now),
    )
    return {"id": image_id, "chapter_id": chapter_id, "filename": filename,
            "title": display_title, "ai_description": "", "created_at": now}

@app.get("/api/chapter-images/{image_id}")
async def get_chapter_image(image_id: str):
    db = get_db()
    row = db.execute_one("SELECT * FROM chapter_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    return dict(row)

@app.get("/api/chapter-images/{image_id}/file")
async def serve_chapter_image(image_id: str):
    db = get_db()
    row = db.execute_one("SELECT * FROM chapter_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(str(path))

class ImageUpdateBody(BaseModel):
    title: Optional[str] = None
    ai_description: Optional[str] = None

@app.put("/api/chapter-images/{image_id}")
async def update_chapter_image(image_id: str, body: ImageUpdateBody):
    db = get_db()
    # Only update fields that were explicitly provided
    updates = {}
    if body.title is not None:
        updates["title"] = body.title
    if body.ai_description is not None:
        updates["ai_description"] = body.ai_description
    if updates:
        set_clause = ", ".join(f"{k}=?" for k in updates)
        db.execute(
            f"UPDATE chapter_images SET {set_clause} WHERE id=?",
            (*updates.values(), image_id),
        )
    return {"ok": True}

@app.delete("/api/chapter-images/{image_id}")
async def delete_chapter_image(image_id: str):
    import os
    db = get_db()
    row = db.execute_one("SELECT file_path FROM chapter_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        os.remove(row["file_path"])
    except FileNotFoundError:
        pass
    db.execute("DELETE FROM chapter_images WHERE id=?", (image_id,))
    return {"ok": True}

@app.post("/api/chapter-images/{image_id}/describe")
async def describe_chapter_image(image_id: str, language: str = Query(default="en")):
    """Use the configured vision model to analyze and describe the image."""
    from norvel_writer.config.settings import get_config
    cfg = get_config()
    vision_model = cfg.vision_model.strip()
    if not vision_model:
        raise HTTPException(status_code=400, detail="No vision model configured. Set OLLAMA_VISION_MODEL in Settings → LLM Config.")

    db = get_db()
    row = db.execute_one("SELECT * FROM chapter_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")

    from norvel_writer.llm.ollama_client import get_client
    from norvel_writer.llm.prompt_builder import _lang_display
    lang = _lang_display(language)
    description = await get_client().describe_image(path, vision_model, language=lang)
    db.execute(
        "UPDATE chapter_images SET ai_description=? WHERE id=?",
        (description, image_id),
    )
    return {"description": description}


# ── Project-level Image Memory ─────────────────────────────────────────────

def _project_images_dir(project_id: str) -> Path:
    from norvel_writer.config.settings import get_config
    d = get_config().project_images_path / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d

@app.get("/api/projects/{project_id}/images")
async def list_project_images(project_id: str):
    db = get_db()
    rows = db.execute(
        "SELECT * FROM project_images WHERE project_id=? ORDER BY created_at ASC",
        (project_id,),
    )
    return [dict(r) for r in rows]

@app.post("/api/projects/{project_id}/images")
async def upload_project_image(project_id: str, file: UploadFile, title: str = Form(default="")):
    import shutil, uuid
    from datetime import datetime, timezone

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {ext}")

    image_id = str(uuid.uuid4())
    filename  = f"{image_id}{ext}"
    dest      = _project_images_dir(project_id) / filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    now = datetime.now(timezone.utc).isoformat()
    display_title = title.strip() or file.filename or filename
    db = get_db()
    db.execute(
        "INSERT INTO project_images(id, project_id, filename, title, ai_description, file_path, created_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (image_id, project_id, filename, display_title, "", str(dest), now),
    )
    return {"id": image_id, "project_id": project_id, "filename": filename,
            "title": display_title, "ai_description": "", "created_at": now}

@app.get("/api/project-images/{image_id}")
async def get_project_image(image_id: str):
    db = get_db()
    row = db.execute_one("SELECT * FROM project_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    return dict(row)

@app.get("/api/project-images/{image_id}/file")
async def serve_project_image(image_id: str):
    db = get_db()
    row = db.execute_one("SELECT * FROM project_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")
    return FileResponse(str(path))

class ProjectImageUpdateBody(BaseModel):
    title: Optional[str] = None
    ai_description: Optional[str] = None

@app.put("/api/project-images/{image_id}")
async def update_project_image(image_id: str, body: ProjectImageUpdateBody):
    db = get_db()
    updates = {}
    if body.title is not None:
        updates["title"] = body.title
    if body.ai_description is not None:
        updates["ai_description"] = body.ai_description
    if updates:
        set_clause = ", ".join(f"{k}=?" for k in updates)
        db.execute(
            f"UPDATE project_images SET {set_clause} WHERE id=?",
            (*updates.values(), image_id),
        )
    return {"ok": True}

@app.delete("/api/project-images/{image_id}")
async def delete_project_image(image_id: str):
    import os
    db = get_db()
    row = db.execute_one("SELECT file_path FROM project_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        os.remove(row["file_path"])
    except FileNotFoundError:
        pass
    db.execute("DELETE FROM project_images WHERE id=?", (image_id,))
    return {"ok": True}

@app.post("/api/project-images/{image_id}/describe")
async def describe_project_image(image_id: str, language: str = Query(default="en")):
    """Use the configured vision model to analyze and describe the project-level image."""
    from norvel_writer.config.settings import get_config
    cfg = get_config()
    vision_model = cfg.vision_model.strip()
    if not vision_model:
        raise HTTPException(status_code=400, detail="No vision model configured.")

    db = get_db()
    row = db.execute_one("SELECT * FROM project_images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file missing on disk")

    from norvel_writer.llm.ollama_client import get_client
    from norvel_writer.llm.prompt_builder import _lang_display
    lang = _lang_display(language)
    description = await get_client().describe_image(path, vision_model, language=lang)
    db.execute(
        "UPDATE project_images SET ai_description=? WHERE id=?",
        (description, image_id),
    )
    return {"description": description}


# ── Chapter summary ────────────────────────────────────────────────────────

@app.get("/api/chapters/{chapter_id}/summary")
async def summarise_chapter(chapter_id: str, language: str = Query(default="en")):
    try:
        pm = get_pm()
        draft = pm.get_accepted_draft(chapter_id)
        raw_content = draft["content"] if draft else ""
        # Strip HTML tags — the editor stores innerHTML; LLMs need plain text
        content = _strip_html(raw_content)
        if not content.strip():
            return {"summary": ""}
        from norvel_writer.core.draft_engine import DraftEngine
        engine = DraftEngine(project_manager=pm)
        summary = await engine.summarise_chapter(content, language=language)
        return {"summary": summary}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Chapter beats ──────────────────────────────────────────────────────────

@app.get("/api/chapters/{chapter_id}/beats")
async def get_beats(chapter_id: str):
    try:
        from norvel_writer.storage.repositories.project_repo import ProjectRepo
        from norvel_writer.storage.db import get_db
        row = ProjectRepo(get_db()).get_chapter(chapter_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Chapter not found")
        return {"beats": row.get("beats") or ""}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class BeatsUpdate(BaseModel):
    beats: str


@app.put("/api/chapters/{chapter_id}/beats")
async def save_beats(chapter_id: str, body: BeatsUpdate):
    try:
        get_pm().update_chapter(chapter_id, beats=body.beats)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class BeatsGenRequest(BaseModel):
    description: str
    language: str = "en"


@app.post("/api/chapters/{chapter_id}/beats/generate")
async def generate_beats(chapter_id: str, body: BeatsGenRequest):
    async def _gen():
        try:
            from norvel_writer.llm.langchain_bridge import chat_stream
            from norvel_writer.llm.prompt_builder import _lang_display
            lang = _lang_display(body.language)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a story structure expert. "
                        "Read the chapter and identify its KEY story beats (up to 15) — the major turning points, "
                        "revelations, decisions, or emotional shifts that drive the plot forward. "
                        "Ignore minor actions, dialogue details, and scene logistics. "
                        "Each beat is ONE short sentence capturing WHAT CHANGES or MATTERS most. "
                        f"You MUST write every beat in {lang}. "
                        "Output ONLY a numbered list, no other text. Format:\n1. [beat]\n2. [beat]\n..."
                    ),
                },
                {"role": "user", "content": f"Identify the key story beats in this chapter:\n\n{body.description}"},
            ]
            async for chunk in await chat_stream(messages):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    return StreamingResponse(_gen(), media_type="text/event-stream")


# ── Style ──────────────────────────────────────────────────────────────────

@app.get("/api/projects/{project_id}/style")
async def get_style(project_id: str):
    try:
        profile = get_pm().get_active_style_profile(project_id)
        return profile or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/projects/{project_id}/style/build")
async def build_style(project_id: str):
    async def _gen():
        try:
            from norvel_writer.core.style_profile import StyleProfileEngine

            engine = StyleProfileEngine()

            # Use the project's content language for style analysis output
            proj = get_pm().get_project(project_id)
            proj_lang = (proj.get("language") or "en") if proj else "en"

            def _progress(pct: int):
                pass  # progress_cb is sync callback; we'll report done at end

            profile_id = await engine.build_profile(
                project_id=project_id,
                language=proj_lang,
                progress_cb=_progress,
            )
            profile = get_pm().get_active_style_profile(project_id)
            yield f"data: {json.dumps({'profile_id': profile_id, 'done': True, 'profile': profile or {}})}\n\n"
        except Exception as exc:
            log.error("build_style error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ── Persona ────────────────────────────────────────────────────────────────

@app.get("/api/projects/{project_id}/persona")
async def get_persona(project_id: str):
    try:
        proj = get_pm().get_project(project_id)
        if proj is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return {"persona": proj.get("persona") or ""}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class PersonaUpdate(BaseModel):
    persona: str


@app.put("/api/projects/{project_id}/persona")
async def save_persona(project_id: str, body: PersonaUpdate):
    try:
        get_pm().update_project(project_id, persona=body.persona)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Settings ───────────────────────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    try:
        from norvel_writer.config.settings import get_config
        cfg = get_config()
        return {
            "ollama_base_url": cfg.ollama_base_url,
            "default_chat_model": cfg.default_chat_model,
            "default_embed_model": cfg.default_embed_model,
            "vision_model": cfg.vision_model,
            "default_content_language": cfg.default_content_language,
            "default_project_language": cfg.default_project_language,
            "theme": cfg.theme,
            "ui_language": cfg.ui_language,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/settings")
async def update_settings(body: Dict[str, Any]):
    try:
        from norvel_writer.config.settings import get_config, set_config, AppConfig
        cfg = get_config()
        updated = cfg.model_copy(update=body)
        set_config(updated)
        updated.save()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── LLM provider config (.env) ────────────────────────────────────────────────

@app.get("/api/llm/config")
async def get_llm_config():
    """Return the current .env path and raw content."""
    try:
        from norvel_writer.llm.langchain_bridge import ensure_env_exists, find_env_path
        path = ensure_env_exists()
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        return {
            "path": str(path),
            "content": raw,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.put("/api/llm/config")
async def save_llm_config(body: Dict[str, Any]):
    """
    Write .env.  Body: { "content": "<full env text>" }
    After saving, resets the LLM singletons so the next request picks up changes.
    """
    try:
        from norvel_writer.llm.langchain_bridge import env_dest, reset_singletons
        dest = env_dest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body.get("content", ""), encoding="utf-8")
        reset_singletons()
        return {"ok": True, "path": str(dest)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
