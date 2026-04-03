"""
Role definition loader for Norvel Writer chat roles.

Roles are defined in TOML files.  The loader checks two locations in order:
  1. <user data dir>/roles/<role>.toml   — user's custom overrides
  2. <package>/resources/roles/<role>.toml — bundled defaults

This means users can copy a default file to their data dir and edit it freely
without touching the installed package.  Changes take effect on the next chat
message with no restart required (files are re-read each call).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)

try:
    import tomllib          # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


def _user_roles_dir() -> Path:
    from norvel_writer.config.settings import get_config
    return get_config().data_dir / "roles"


def _bundled_roles_dir() -> Path:
    return Path(__file__).parent.parent / "resources" / "roles"


def load_role(role: str) -> Dict[str, Any]:
    """
    Load and return the TOML data for a role ('editor', 'writer', 'qa').
    Returns an empty dict if no file is found (engine falls back to hard-coded defaults).
    """
    filename = f"{role}.toml"
    candidates = [
        _user_roles_dir() / filename,
        _bundled_roles_dir() / filename,
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                log.debug("Loaded role %r from %s", role, path)
                return data
            except Exception as exc:
                log.warning("Failed to parse role file %s: %s", path, exc)
    log.debug("No role file found for %r — using built-in defaults", role)
    return {}


def role_identity(role: str) -> Dict[str, str]:
    """Return the [identity] section: name, hint, background."""
    return load_role(role).get("identity", {})


def role_hint(role: str) -> str:
    """Return the one-line hint string for the UI."""
    return role_identity(role).get("hint", "")


def list_user_roles_dir() -> Path:
    """Return the user-editable roles directory (creating it if needed)."""
    d = _user_roles_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_user_role_files() -> None:
    """
    Copy bundled role files to the user data dir if they don't exist yet.
    Called at startup so users always have editable copies to hand.
    """
    import shutil
    user_dir = list_user_roles_dir()
    bundled_dir = _bundled_roles_dir()
    for src in bundled_dir.glob("*.toml"):
        dest = user_dir / src.name
        if not dest.exists():
            try:
                shutil.copy2(src, dest)
                log.info("Copied default role file to %s", dest)
            except Exception as exc:
                log.warning("Could not copy role file %s: %s", src.name, exc)
