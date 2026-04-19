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

### Read Aloud

- **Read Aloud** — reads the current chapter aloud using Microsoft neural TTS (edge-tts); select Male or Female voice from the topbar before clicking ▶ Read
- **Auto-language** — voice is chosen from the **AI Language** selector; 18 languages supported (English, Chinese, Japanese, Korean, French, German, Spanish, and more)
- **No extra install** — edge-tts is included in the package; requires an **internet connection** at read time (audio is synthesised by Microsoft's free Azure neural TTS service)

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

1. **Ollama** — download from [ollama.com](https://ollama.com/download)

### Windows (recommended)

```bat
run.bat
```

That's it. `run.bat` auto-installs [uv](https://astral.sh/uv), downloads Python 3.11, and installs all packages on first launch.

### Manual install

```bash
# Clone
git clone <repo>
cd norvel-writer

# Install with uv (recommended — manages Python version automatically)
uv sync
uv run python -m norvel_writer

# Or with pip (requires Python 3.11 already installed)
pip install -e .
python -m norvel_writer
```

> **Windows build error — `chroma-hnswlib` / "Microsoft Visual C++ 14.0 required"**
>
> This happens when `uv sync` or `pip install` picks up Python 3.12+ instead of 3.11.
> Pre-built wheels for `chroma-hnswlib` only exist for Python 3.11 on Windows.
> The repo includes a `.python-version` file that pins uv to 3.11 automatically.
> If you see this error, make sure you are running `uv sync` (not `pip install`) from the repo root,
> or use `run.bat` which always passes `--python 3.11`.

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

### Ollama Advanced Generation Settings

When `LLM_PROVIDER=ollama`, the **Settings** dialog exposes an **Ollama Advanced Generation Settings** panel with per-request sampling controls. These sliders directly influence every story generation call — they are not cosmetic.

| Parameter | Default | Range | Effect |
|---|---|---|---|
| Temperature | 0.85 | 0.0 – 2.0 | Higher = more creative, lower = more stable |
| Top-p | 0.90 | 0.0 – 1.0 | Higher = more token variety, lower = more conservative |
| Min-p | 0.03 | 0.0 – 1.0 | Filters out very unlikely token candidates |
| Repeat penalty | 1.08 | 0.8 – 2.0 | Reduces repetitive wording and looping |
| Seed | (empty) | integer | Same seed can help reproduce similar outputs |
| num_ctx | (from .env) | ≥ 512 | Context window size; overrides `OLLAMA_NUM_CTX` |
| num_predict | (from .env) | ≥ 64 | Max tokens per call; overrides `OLLAMA_NUM_PREDICT` |

Three one-click **presets** are available in the panel:

| Preset | Temperature | Top-p | Min-p | Repeat penalty | Best for |
|---|---|---|---|---|---|
| **Stable** | 0.60 | 0.85 | 0.05 | 1.15 | Clean continuity, less surprising output |
| **Balanced** | 0.85 | 0.90 | 0.03 | 1.08 | Normal day-to-day writing |
| **Creative** | 1.10 | 0.95 | 0.02 | 1.05 | More surprising output, some consistency trade-off |

Settings are saved to `config.toml` and restored on restart. They override the corresponding `.env` variables when set. If the panel is hidden (provider is not Ollama), values are preserved but not sent.

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

## Getting Professional, On-Style Results

The app has several layered mechanisms for teaching the AI your desired voice and quality standard. Use them together for the best results.

### 1. Upload Style Samples (strongest signal)

Go to **Documents → Style tab** and upload 1–3 chapters of prose whose style you want to match — a favourite author, your own best work, or a mix of both. These are stored in a dedicated style collection and injected into every generation as Priority 6.

**Tips:**
- One 3 000-word passage beats ten 300-word fragments
- Mix dialogue-heavy and narrative-heavy samples for balanced coverage
- For cloud / large-context models (`CONTEXT_MODE=full`), the full text of every style document is sent in each call — no chunking loss

### 2. Write a Persona (permanent voice instructions)

In **Project Settings → Persona**, describe your writing voice like you would brief a ghostwriter. This is Priority 2 — it overrides style samples and all stylistic defaults.

```
Write in close third-person past tense. Sentences are mid-length with
occasional short punches for impact. No adverbs on dialogue tags —
use action beats instead. Inner monologue in italics. Prose is lean
and cinematic, never flowery. Tension should simmer beneath the surface
even in quieter scenes.
```

### 3. Pin Editor Notes (per-chapter persistent rules)

Click the **📌 Editor** button in the chat panel and write chapter-specific style or quality rules. These persist across sessions and are injected at Priority 3 into every generation for that chapter.

```
This chapter should feel tense and claustrophobic. Short paragraphs.
Avoid passive voice. Every scene must end on an unresolved beat.
```

### 4. Create a Style Rules document

Add a **Notes** document to your project (e.g. `Writing Style Guide`) listing your prose dos and don'ts. In cloud / full-doc mode the entire document is sent to the AI on every call.

```markdown
## Prose Rules
- Show emotion through action, not adjectives
- Use sensory detail: what the character smells, hears, and feels
- Dialogue subtext: characters rarely say exactly what they mean

## Forbidden patterns
- No adverbs attached to dialogue tags (said softly, shouted angrily)
- No dream sequences unless explicitly listed in beats
- No word-count summaries or chapter epilogue narration
```

### 5. Be specific in the Draft Instruction field

Instead of a vague prompt, give the AI concrete direction:

> *Write the chapter in the style of Cormac McCarthy's The Road — sparse prose, no quotation marks on dialogue, short declarative sentences, a tone of bleak but resilient humanity. Each beat should expand to at least 600 words.*

### 6. Use the QA → Editor feedback loop

After a generation you are not satisfied with:
1. Open **Chat → QA role** and ask it to audit the prose quality
2. Pin the QA report with 📌
3. Regenerate — the next call sees the pinned issues and avoids them

### Priority order (all AI entry points)

Every generation honours this strict order:

| Priority | Source | Where to set it |
|---|---|---|
| 1 | Current instruction | Draft field or chat message |
| 2 | **Author Persona** | Project Settings |
| 3 | **Pinned Editor Notes** | 📌 in Chat panel |
| 4 | Codex / Beats / Notes / Research / Images | Documents tab |
| 5 | Pinned QA Notes | 📌 in Chat panel |
| 6 | **Style Samples** | Documents → Style tab |

### Recommended setup for a new project

1. Write a **Persona** (100–200 words describing your exact voice)
2. Upload **2 style sample chapters** from your reference author(s)
3. Create a **Notes doc** called "Style Rules" with your prose dos/don'ts
4. Generate one chapter, then use **Chat → Editor** to critique it and pin the result
5. Regenerate — the second attempt will be noticeably closer to your target

The Persona + Style Samples combination typically covers 80 % of the style gap. Pinned Editor Notes handle the remaining chapter-specific 20 %.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit/          # unit tests (no Ollama needed)
pytest tests/integration/   # requires running Ollama
```
