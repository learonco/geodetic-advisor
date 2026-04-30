"""LLM factory for selecting the appropriate chat model.

When a Gemini API key is supplied, ``ChatGoogleGenerativeAI`` is returned.
When no key (or an empty string) is provided, ``ChatOllama`` (local) is
returned instead, using the ``OLLAMA_BASE_URL`` environment variable as the
base URL (default: ``http://localhost:11434``).
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

GEMINI_MODEL = "models/gemini-2.0-flash"
OLLAMA_MODEL = "gemma3"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"


def build_llm(gemini_api_key: str | None) -> BaseChatModel:
    """Build the chat LLM based on the availability of a Gemini API key.

    Args:
        gemini_api_key: Gemini API key string. ``None`` or an empty string
            will cause an Ollama local model to be used instead.

    Returns:
        A configured ``BaseChatModel`` instance — either
        ``ChatGoogleGenerativeAI`` or ``ChatOllama``.
    """
    if gemini_api_key:
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=gemini_api_key,
        )

    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=base_url,
        temperature=0,
    )
