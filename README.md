# Norvel Writer

Local-first writing assistant powered by [Ollama](https://ollama.com). Your writing stays on your device.

## Features

- **Local AI** — all drafting and revision runs on-device via Ollama
- **Project memory** — ingest codex, beats, character sheets, prior chapters (.txt, .md, .docx, .pdf, .json)
- **Style adaptation** — upload sample texts and build an author style profile
- **AI drafting** — continue, rewrite, expand passages with RAG context
- **External edit round-trip** — export → edit in any tool → re-import → continue
- **Multilingual** — write in any language Ollama supports
- **NotebookLM integration** — export project materials in NotebookLM-ready format
- **Local-only** — no cloud APIs required for core features

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

### Recommended models

| Role | Model | VRAM |
|---|---|---|
| Chat | `llama3.2:3b` | ~4 GB |
| Chat (better) | `llama3.1:8b` | ~8 GB |
| Embeddings | `nomic-embed-text` | ~300 MB |

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
├── ui/                     # PySide6 desktop UI
│   ├── main_window.py
│   ├── panels/             # Project, Editor, Draft, Memory, Style
│   ├── setup_wizard/       # First-run wizard
│   ├── dialogs/
│   └── widgets/
├── utils/                  # Chunker, async bridge, text utils
└── resources/prompts/      # Jinja2 prompt templates
```

## Architecture

- **UI**: PySide6 (Qt for Python)
- **Inference**: Ollama Python SDK (local, no cloud)
- **Vectors**: ChromaDB embedded (no server)
- **Database**: SQLite with WAL mode
- **Embeddings**: `nomic-embed-text` via Ollama
- **Async bridge**: persistent asyncio loop on QThread

## Development

```bash
pip install -e ".[dev]"
pytest tests/unit/          # unit tests (no Ollama needed)
pytest tests/integration/   # requires running Ollama
```
