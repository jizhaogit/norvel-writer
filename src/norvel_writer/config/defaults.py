"""Hard-coded application defaults."""

APP_NAME = "NorvelWriter"
APP_AUTHOR = "NorvelWriter"

# Ollama
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_MODEL = "gemma3:4b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
FALLBACK_CHAT_MODELS = [
    "gemma3:4b",
    "gemma3:1b",
    "llama3.2:3b",
    "llama3.1:8b",
    "mistral:7b",
    "phi3:mini",
]
RECOMMENDED_EMBED_MODEL = "nomic-embed-text"
EMBED_DIMENSION = 768  # nomic-embed-text default

# Chunking
DEFAULT_CHUNK_TOKENS = 512
DEFAULT_CHUNK_OVERLAP = 64

# RAG
DEFAULT_RAG_TOP_K = 8
DEFAULT_STYLE_TOP_K = 4

# Autosave
AUTOSAVE_INTERVAL_MS = 30_000  # 30 seconds

# Supported file formats
SUPPORTED_FORMATS = {".txt", ".md", ".docx", ".pdf", ".json"}
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
ALL_SUPPORTED_FORMATS = SUPPORTED_FORMATS | IMAGE_FORMATS

# Vision models known to support image input in Ollama
KNOWN_VISION_MODELS = [
    "llava:7b",
    "llava:13b",
    "llava:34b",
    "llama3.2-vision",
    "llama3.2-vision:11b",
    "minicpm-v",
    "moondream",
    "bakllava",
]
DEFAULT_VISION_MODEL = "llava:7b"

# ── Language registry ──────────────────────────────────────────────────────
# Maps ISO code → (display label, native label)
# Used by settings dialog, draft panel, chat panel, and new-project dialog.
LANGUAGES: dict[str, tuple[str, str]] = {
    "en":    ("English",             "English"),
    "zh":    ("Chinese",             "中文"),
    "zh-tw": ("Chinese Traditional", "繁體中文"),
    "ja":    ("Japanese",            "日本語"),
    "ko":    ("Korean",              "한국어"),
    "es":    ("Spanish",             "Español"),
    "fr":    ("French",              "Français"),
    "de":    ("German",              "Deutsch"),
    "ru":    ("Russian",             "Русский"),
    "pt":    ("Portuguese",          "Português"),
    "ar":    ("Arabic",              "العربية"),
    "hi":    ("Hindi",               "हिन्दी"),
    "it":    ("Italian",             "Italiano"),
    "nl":    ("Dutch",               "Nederlands"),
    "pl":    ("Polish",              "Polski"),
    "tr":    ("Turkish",             "Türkçe"),
    "vi":    ("Vietnamese",          "Tiếng Việt"),
    "th":    ("Thai",                "ภาษาไทย"),
}

# Full display label used in prompts, e.g. "Chinese (中文)"
def language_display(code: str) -> str:
    """Return 'English Name (Native Name)' for use in AI prompts.

    Normalises langdetect variant codes (zh-cn, pt-br, etc.) to our registry
    keys before lookup so the AI always receives a proper language name rather
    than a raw ISO code like 'zh-cn'.
    """
    _aliases: dict = {
        "zh-cn": "zh",
        "zh-tw": "zh-tw",
        "pt-br": "pt",
        "pt-pt": "pt",
        "he":    "he",   # Hebrew — not in registry, returned as-is below
    }
    code = _aliases.get(code, code)
    if code not in LANGUAGES:
        return code   # last resort: return raw code
    eng, native = LANGUAGES[code]
    return f"{eng} ({native})" if eng != native else eng

DEFAULT_CONTENT_LANGUAGE = "en"

# UI locale codes for Qt translations (future i18n)
UI_LOCALES: dict[str, str] = {
    "en":    "en_US",
    "zh":    "zh_CN",
    "zh-tw": "zh_TW",
    "ja":    "ja_JP",
    "ko":    "ko_KR",
    "es":    "es_ES",
    "fr":    "fr_FR",
    "de":    "de_DE",
    "ru":    "ru_RU",
    "pt":    "pt_BR",
    "ar":    "ar_SA",
    "hi":    "hi_IN",
    "it":    "it_IT",
    "nl":    "nl_NL",
    "pl":    "pl_PL",
    "tr":    "tr_TR",
    "vi":    "vi_VN",
    "th":    "th_TH",
}

# Memory doc types
DOC_TYPES = ["codex", "beats", "style_sample", "draft", "research", "notes", "visual"]

# Style modes
STYLE_MODES = [
    "imitate_closely",
    "inspired_by",
    "preserve_tone_rhythm",
    "avoid_exact_phrasing",
]
