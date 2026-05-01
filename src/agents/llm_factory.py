"""LLM factory for selecting the appropriate chat model.

The ``provider`` argument drives model selection — not the presence or absence
of a key.  Pass ``"gemini"`` to use Google Gemini, or ``"ollama"`` for a local
Ollama model.  Credentials are supplied separately via ``gemini_api_key`` and
``ollama_url``; neither credential implicitly selects the provider.

For Ollama, the effective base URL is resolved in priority order:
    1. The explicit ``ollama_url`` argument.
    2. The ``OLLAMA_BASE_URL`` environment variable.
    3. The hardcoded default ``http://localhost:11434``.
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

GEMINI_MODEL = "models/gemini-3-flash-preview"
OLLAMA_MODEL = "mistral-nemo:12b"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


def build_llm(
    provider: str,
    gemini_api_key: str | None = None,
    ollama_url: str | None = None,
) -> BaseChatModel:
    """Build the chat LLM for the given provider.

    Args:
        provider: Which LLM backend to use.  Must be ``"gemini"`` or
            ``"ollama"``.
        gemini_api_key: Google Gemini API key.  Only used when
            ``provider="gemini"``.
        ollama_url: Base URL for the Ollama server.  Only used when
            ``provider="ollama"``.  Falls back to the ``OLLAMA_BASE_URL``
            environment variable, then ``http://localhost:11434``.

    Returns:
        A configured ``BaseChatModel`` instance — either
        ``ChatGoogleGenerativeAI`` or ``ChatOllama``.
    """
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=gemini_api_key,
        )

    base_url = ollama_url or os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=base_url,
        temperature=0,
    )
