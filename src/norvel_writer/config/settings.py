"""Application configuration via pydantic-settings with TOML persistence."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import tomli_w
from platformdirs import user_data_dir, user_config_dir
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from norvel_writer.config.defaults import (
    APP_AUTHOR,
    APP_NAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONTENT_LANGUAGE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_URL,
)

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def _default_data_dir() -> Path:
    return Path(user_data_dir(APP_NAME, APP_AUTHOR))


def _default_config_path() -> Path:
    config_dir = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.toml"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NORVEL_", extra="ignore")

    # Paths
    data_dir: Path = Field(default_factory=_default_data_dir)

    # Ollama
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    default_chat_model: str = DEFAULT_CHAT_MODEL
    default_embed_model: str = DEFAULT_EMBED_MODEL

    # UI
    ui_language: str = "en"          # language code for the app interface (requires restart)
    theme: str = "dark"

    # Language defaults — used throughout the app as fallback
    default_content_language: str = DEFAULT_CONTENT_LANGUAGE   # language for AI-generated content
    default_project_language: str = DEFAULT_CONTENT_LANGUAGE   # pre-filled when creating new projects

    # Vision model for image description (optional — leave empty to skip vision processing)
    vision_model: str = ""  # e.g. "llava:7b" or "llama3.2-vision"

    # State
    first_run_complete: bool = False
    last_opened_project_id: Optional[str] = None

    @field_validator("data_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: object) -> Path:
        return Path(v)  # type: ignore[arg-type]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "norvel_writer.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def projects_path(self) -> Path:
        return self.data_dir / "projects"

    @property
    def logs_path(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for p in [
            self.data_dir,
            self.chroma_path,
            self.projects_path,
            self.logs_path,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def save(self, config_path: Optional[Path] = None) -> None:
        path = config_path or _default_config_path()
        data = {
            "data_dir": str(self.data_dir),
            "ollama_base_url": self.ollama_base_url,
            "default_chat_model": self.default_chat_model,
            "default_embed_model": self.default_embed_model,
            "ui_language": self.ui_language,
            "theme": self.theme,
            "default_content_language": self.default_content_language,
            "default_project_language": self.default_project_language,
            "vision_model": self.vision_model,
            "first_run_complete": self.first_run_complete,
            "last_opened_project_id": self.last_opened_project_id,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        path = config_path or _default_config_path()
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            return cls(**data)
        return cls()


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config


def set_config(config: AppConfig) -> None:
    global _config
    _config = config
