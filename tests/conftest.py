"""Pytest configuration and shared fixtures."""
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from norvel_writer.storage.db import init_db
from norvel_writer.config.settings import AppConfig, set_config


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def app_config(tmp_data_dir: Path) -> Generator[AppConfig, None, None]:
    cfg = AppConfig(
        data_dir=tmp_data_dir,
        ollama_base_url="http://localhost:11434",
        default_chat_model="llama3.2:3b",
        default_embed_model="nomic-embed-text",
        first_run_complete=True,
    )
    cfg.ensure_dirs()
    set_config(cfg)
    yield cfg


@pytest.fixture
def db(app_config: AppConfig):
    return init_db(app_config.db_path)


@pytest.fixture
def project_id(db) -> str:
    from norvel_writer.storage.repositories.project_repo import ProjectRepo
    repo = ProjectRepo(db)
    return repo.create_project("Test Project", language="en")
