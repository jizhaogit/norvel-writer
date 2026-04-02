"""Jinja2-based prompt builder. Templates live in resources/prompts/."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_TEMPLATE_DIR: Optional[Path] = None


def _get_template_dir() -> Path:
    global _TEMPLATE_DIR
    if _TEMPLATE_DIR is None:
        # Try package resources first
        here = Path(__file__).parent.parent
        candidate = here / "resources" / "prompts"
        if candidate.exists():
            _TEMPLATE_DIR = candidate
        else:
            # PyInstaller bundle: sys._MEIPASS
            import sys
            if hasattr(sys, "_MEIPASS"):
                _TEMPLATE_DIR = Path(sys._MEIPASS) / "resources" / "prompts"  # type: ignore
            else:
                _TEMPLATE_DIR = candidate
    return _TEMPLATE_DIR


def _env():
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    template_dir = _get_template_dir()
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_template(name: str, **kwargs: Any) -> str:
    """Render a Jinja2 template from resources/prompts/."""
    env = _env()
    try:
        tmpl = env.get_template(name)
        return tmpl.render(**kwargs)
    except Exception as exc:
        log.error("Failed to render template %r: %s", name, exc)
        raise


def build_continuation_messages(
    current_text: str,
    rag_chunks: List[str],
    style_chunks: List[str],
    style_profile: Optional[Dict[str, Any]],
    user_instruction: str,
    language: str,
    style_mode: str,
    constraints: Optional[List[str]] = None,
    persona: str = "",
    beats: str = "",
) -> List[Dict[str, str]]:
    """Build the messages list for a continuation request."""
    system_prompt = render_template(
        "continue_draft.j2",
        rag_chunks=rag_chunks,
        style_chunks=style_chunks,
        style_profile=style_profile,
        language=language,
        style_mode=style_mode,
        constraints=constraints or [],
        persona=persona,
        beats=beats,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{user_instruction}\n\n---\n{current_text}",
        },
    ]
    return messages


def build_rewrite_messages(
    passage: str,
    rag_chunks: List[str],
    style_chunks: List[str],
    style_profile: Optional[Dict[str, Any]],
    user_instruction: str,
    language: str,
    style_mode: str,
    persona: str = "",
) -> List[Dict[str, str]]:
    system_prompt = render_template(
        "rewrite_passage.j2",
        rag_chunks=rag_chunks,
        style_chunks=style_chunks,
        style_profile=style_profile,
        language=language,
        style_mode=style_mode,
        persona=persona,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{user_instruction}\n\n---\n{passage}",
        },
    ]
    return messages


def build_style_extraction_messages(
    sample_texts: List[str],
    model_language: str = "en",
) -> List[Dict[str, str]]:
    combined = "\n\n---\n\n".join(sample_texts[:5])  # cap at 5 samples per call
    system_prompt = render_template(
        "extract_style.j2",
        language=model_language,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined},
    ]
