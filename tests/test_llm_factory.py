"""Unit tests for src/agents/llm_factory — TDD red/green cycle."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestBuildLlm:
    """Tests for build_llm() factory function."""

    def test_build_llm_with_key_returns_gemini(self):
        """When provider is 'gemini', a ChatGoogleGenerativeAI instance is returned."""
        mock_gemini = MagicMock()
        mock_cls = MagicMock(return_value=mock_gemini)

        with patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_cls):
            from src.agents import llm_factory
            import importlib
            importlib.reload(llm_factory)

            result = llm_factory.build_llm("gemini", gemini_api_key="test-key-123")

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["google_api_key"] == "test-key-123"
        assert result is mock_gemini

    def test_build_llm_without_key_returns_ollama(self):
        """When provider is 'ollama', a ChatOllama instance is returned."""
        mock_ollama = MagicMock()
        mock_cls = MagicMock(return_value=mock_ollama)

        with patch("langchain_ollama.ChatOllama", mock_cls):
            from src.agents import llm_factory
            import importlib
            importlib.reload(llm_factory)

            result = llm_factory.build_llm("ollama")

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == llm_factory.OLLAMA_MODEL
        assert result is mock_ollama

    def test_build_llm_ollama_uses_env_base_url(self, monkeypatch):
        """OLLAMA_BASE_URL env var is used when no explicit ollama_url is passed."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434")

        mock_ollama = MagicMock()
        mock_cls = MagicMock(return_value=mock_ollama)

        with patch("langchain_ollama.ChatOllama", mock_cls):
            from src.agents import llm_factory
            import importlib
            importlib.reload(llm_factory)

            llm_factory.build_llm("ollama")

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["base_url"] == "http://remote-ollama:11434"

    def test_build_llm_ollama_default_base_url(self, monkeypatch):
        """When OLLAMA_BASE_URL is not set, the default http://localhost:11434 is used."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        mock_cls = MagicMock(return_value=MagicMock())

        with patch("langchain_ollama.ChatOllama", mock_cls):
            from src.agents import llm_factory
            import importlib
            importlib.reload(llm_factory)

            llm_factory.build_llm("ollama")

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:11434"

    def test_build_llm_ollama_uses_explicit_url(self, monkeypatch):
        """An explicit ollama_url takes priority over the OLLAMA_BASE_URL env var."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama:11434")

        mock_cls = MagicMock(return_value=MagicMock())

        with patch("langchain_ollama.ChatOllama", mock_cls):
            from src.agents import llm_factory
            import importlib
            importlib.reload(llm_factory)

            llm_factory.build_llm("ollama", ollama_url="http://custom:11434")

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["base_url"] == "http://custom:11434"

    def test_build_llm_gemini_uses_correct_model(self):
        """Gemini LLM is instantiated with the expected model name."""
        mock_cls = MagicMock(return_value=MagicMock())

        with patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_cls):
            from src.agents import llm_factory
            import importlib
            importlib.reload(llm_factory)

            llm_factory.build_llm("gemini", gemini_api_key="any-key")

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "models/gemini-2.0-flash"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_retries"] == 2

    def test_build_llm_gemini_with_no_key_does_not_use_ollama(self):
        """Provider 'gemini' always returns Gemini — key validation is Gemini's responsibility."""
        mock_gemini_cls = MagicMock(return_value=MagicMock())
        mock_ollama_cls = MagicMock(return_value=MagicMock())

        with patch("langchain_google_genai.ChatGoogleGenerativeAI", mock_gemini_cls), \
             patch("langchain_ollama.ChatOllama", mock_ollama_cls):
            from src.agents import llm_factory
            import importlib
            importlib.reload(llm_factory)

            llm_factory.build_llm("gemini", gemini_api_key=None)

        mock_gemini_cls.assert_called_once()
        mock_ollama_cls.assert_not_called()
