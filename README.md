# Norvel Writer

Local-first AI writing assistant powered by [Ollama](https://ollama.com). Your writing stays on your device — no cloud, no subscriptions, no data leaving your machine.

## Features

### Core Writing

- **AI Drafting** — continue, expand, or rewrite passages with a single click; Draft AI shares the same full context stack as the Chat Writer
- **Chat Writer** — conversational AI writing assistant with persistent per-chapter chat history
- **Rewrite Versions** — generate multiple rewrite candidates and compare them before accepting
- **Rich Text Editor** — in-browser editor with word count tracking and auto-save

### Context & Memory

- **Project Codex** — ingest reference documents (character sheets, world-building, prior chapters) in `.txt`, `.md`, `.docx`, `.pdf`, `.json`
- **Chapter Beats** — per-chapter story beat notes fed directly into every AI prompt
- **Pinned Editor Notes** — persistent per-chapter guidance for the Writer AI (pinned across sessions, never cleared by chat reset)
- **Pinned QA Notes** — persistent per-chapter quality/consistency rules for the QA AI
- **RAG Context** — ChromaDB vector search retrieves the most relevant codex passages for every generation

### Image Memory

- **Chapter Image Gallery** — attach reference images to individual chapters (character art, location photos, mood references)
- **Project Image Memory** — project-wide visual reference library for global assets (maps, cast portraits, mood boards)
- **AI Image Description** — optionally describe any image using a local vision model (e.g. `llava`, `llama3.2-vision`); descriptions are saved to the database
- **Visual Context in AI** — saved image descriptions are automatically injected into Writer, Editor, and QA prompts at the same priority level as Codex

### Author Voice & Style

- **Author Persona** — define your narrative voice once at the project level; the Persona overrides all stylistic defaults and takes priority over style samples
- **Style Profiles** — upload sample writing and let the AI build a detailed style profile (rhythm, sentence structure, tone, vocabulary)
- **Style Samples** — raw reference excerpts used as secondary stylistic guidance, automatically deferred to the Persona when one is set
- **Unified Priority Stack** — all AI entry points (Chat Writer, Draft Continue, Draft Rewrite) follow the same strict prompt priority order:
  1. Immediate user request
  2. Author Persona (highest voice authority)
  3. Editor Notes (pinned guidance)
  4. Codex + beats + research + visual descriptions
  5. QA Notes (consistency rules)
  6. Style samples (secondary — defers to Persona)
  7. Existing draft context

### AI Roles

- **Writer** — drafts and continues prose with full project context
- **Editor** — reviews and polishes text with pinned editorial guidance
- **QA** — checks consistency against codex, beats, and pinned QA rules

### Project Management

- **Multi-project** — manage multiple books/projects in a single app
- **Chapter Management** — create, reorder, rename, and delete chapters from the sidebar
- **Chapter Sidebar** — quick-access chapter list with inline rename (✏) and delete (🗑) controls
- **External Edit Round-Trip** — export a chapter → edit in any external tool → re-import → diff and continue
- **NotebookLM Export** — export project materials in NotebookLM-ready format for additional research workflows

### Internationalisation

- **Multilingual content** — write in any language Ollama supports
- **Per-project language** — set a target language per project; AI generates content in that language
- **UI language** — interface language is configurable separately from content language

### Privacy & Infrastructure

- **100% local** — all inference runs on-device via Ollama; no cloud APIs required
- **No telemetry** — nothing phoned home
- **SQLite + ChromaDB** — all data stored locally in platform-standard directories
- **Vision model optional** — image description requires a vision-capable Ollama model; all other features work without one

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** — download from [ollama.com](https://ollama.com/download)

### Install

```bash
# Clone and install
git clone <repo>
cd norvel-writer
pip install -e .

# Run
python -m norvel_writer
```

On first launch, the setup wizard will guide you through:
- Verifying Ollama is running
- Downloading a chat model (e.g. `llama3.2:3b`)
- Downloading the embedding model (`nomic-embed-text`)

### Recommended Models

| Role | Model | VRAM |
|---|---|---|
| Chat | `llama3.2:3b` | ~4 GB |
| Chat (better) | `llama3.1:8b` | ~8 GB |
| Embeddings | `nomic-embed-text` | ~300 MB |
| Vision (optional) | `llava:7b` | ~8 GB |
| Vision (optional) | `llama3.2-vision` | ~8 GB |

Set your vision model in **Settings → Vision Model** to enable AI image description. Leave it blank to skip vision processing entirely.

## Windows Build

```bat
cd packaging
build_windows.bat
```

Output: `dist\NorvelWriter\NorvelWriter.exe`

## Project Structure

```
src/norvel_writer/
├── app.py                  # Bootstrap
├── config/                 # Settings, migrations
├── core/                   # DraftEngine, StyleProfileEngine, DiffEngine, Exporters
├── ingestion/              # Document loaders + pipeline
├── llm/                    # Ollama client, embedder, prompt builder
├── storage/                # SQLite + ChromaDB + repositories
├── api/                    # FastAPI server (served via embedded browser)
├── web/                    # Single-page HTML/JS/CSS frontend
└── utils/                  # Chunker, async bridge, text utils
```

## Architecture

- **Frontend**: Single-page app (vanilla HTML/JS/CSS) served by FastAPI
- **Backend**: FastAPI with streaming SSE for AI responses
- **Inference**: Ollama Python SDK (local, no cloud)
- **Vectors**: ChromaDB embedded (no server required)
- **Database**: SQLite with WAL mode + schema migrations
- **Embeddings**: `nomic-embed-text` via Ollama
- **Image storage**: Local filesystem under platform data directory

## Configuration

All settings live in `.env` (copy `.env.example` to get started). The app also exposes **Settings → LLM Config** to edit `.env` directly from the UI.

### LLM Provider

| Variable | Default | Options |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` · `openai` · `anthropic` · `gemini` |
| `EMBEDDINGS_PROVIDER` | `ollama` | `ollama` · `openai` |

### Gemma 4 e2b — Context Window Tiers

Gemma 4 e2b supports up to 128 K tokens. Match your settings to your available VRAM:

| Tier / VRAM | `OLLAMA_NUM_CTX` | `OLLAMA_NUM_PREDICT` | RAG budget | Style budget | Text budget |
|---|---|---|---|---|---|
| Tier 1 — 8 GB  | 32 768  | 8 192 | 9 000  | 3 500 | 8 000  |
| Tier 2 — 16 GB | 65 536  | 8 192 | 22 000 | 8 000 | 19 000 |
| Tier 3 — 24 GB+| 131 072 | 8 192 | 48 000 | 18 000| 42 000 |

Recommended Gemma 4 sampling parameters: `OLLAMA_TEMPERATURE=1.0`, `OLLAMA_TOP_P=0.95`, `OLLAMA_TOP_K=64`.

### Context Budgets

These cap how many tokens of each section are included in every prompt (applies to all writers, Editor, QA, and Chat Writer):

| Variable | Default | Purpose |
|---|---|---|
| `CONTEXT_RAG_BUDGET` | `9000` | Codex + beats + notes + research chunks |
| `CONTEXT_STYLE_BUDGET` | `3500` | Style reference sample chunks |
| `CONTEXT_TEXT_BUDGET` | `8000` | Existing chapter draft text |

Budget formula: `input_budget = NUM_CTX − NUM_PREDICT − 1000` (prompt overhead).
Then: RAG ≈ 40 %, Style ≈ 15 %, Text ≈ 35 %, remaining 10 % for beats block / editor note / QA note / persona.

### Codex Distance Threshold (Write from Beats)

In **Write from Beats** mode, codex chunks are retrieved separately and filtered by their cosine distance to the beats query. This prevents large world-building documents from consuming the token budget with irrelevant content.

| `CONTEXT_CODEX_THRESHOLD` | Behaviour |
|---|---|
| `0.35` | Very strict — only chunks that closely match a named character/place in the beats |
| `0.50` | Moderate — relevant details in, noise out ← **recommended default** |
| `0.65` | Permissive — lets in most of the codex; useful for very short beats |
| `1.00` | Off — no filtering (original behaviour) |

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit/          # unit tests (no Ollama needed)
pytest tests/integration/   # requires running Ollama
```
