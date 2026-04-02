"""Application entry point: start FastAPI/uvicorn and open browser."""
from __future__ import annotations
import logging
import sys
import threading
import webbrowser

log = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 7477

def run() -> int:
    from norvel_writer.config.settings import get_config
    cfg = get_config()
    cfg.ensure_dirs()

    from norvel_writer.utils.logging_config import setup_logging
    setup_logging(cfg.logs_path)

    from norvel_writer.storage.db import init_db
    init_db(cfg.db_path)

    # Pre-download NLTK data silently
    try:
        import nltk
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

    # Create .env from template if it doesn't exist yet
    try:
        from norvel_writer.llm.langchain_bridge import ensure_env_exists
        env_path = ensure_env_exists()
        log.info("LLM config: %s", env_path)
    except Exception:
        pass

    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f"http://{HOST}:{PORT}")

    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    from norvel_writer.api.app import app
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    return 0
