"""Unit tests for src/agents/geodetic — TDD red/green cycle.

Tests cover the create_geodetic_agent() factory and the module-level
constants SYSTEM_PROMPT and TOOLS.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestCreateGeodeticAgent:
    """Tests for the create_geodetic_agent() factory."""

    def _import_fresh(self, mock_build_llm=None, mock_create_agent=None):
        """Reload the geodetic module with patches applied."""
        mock_build_llm = mock_build_llm or MagicMock(return_value=MagicMock())
        mock_create_agent = mock_create_agent or MagicMock(return_value=MagicMock())

        with patch("src.agents.llm_factory.build_llm", mock_build_llm), \
             patch("langchain.agents.create_agent", mock_create_agent):
            from src.agents import geodetic
            importlib.reload(geodetic)
            return geodetic, mock_build_llm, mock_create_agent

    def test_create_geodetic_agent_with_key_calls_build_llm_with_key(self):
        """Passing a Gemini key propagates it to build_llm."""
        mock_build = MagicMock(return_value=MagicMock())
        mock_create = MagicMock(return_value=MagicMock())

        with patch("src.agents.llm_factory.build_llm", mock_build), \
             patch("langchain.agents.create_agent", mock_create):
            from src.agents import geodetic
            importlib.reload(geodetic)
            geodetic.create_geodetic_agent(gemini_api_key="my-key")

        mock_build.assert_called_with("my-key")

    def test_create_geodetic_agent_without_key_calls_build_llm_with_none(self):
        """Calling without a key passes None to build_llm."""
        mock_build = MagicMock(return_value=MagicMock())
        mock_create = MagicMock(return_value=MagicMock())

        with patch("src.agents.llm_factory.build_llm", mock_build), \
             patch("langchain.agents.create_agent", mock_create):
            from src.agents import geodetic
            importlib.reload(geodetic)
            geodetic.create_geodetic_agent(gemini_api_key=None)

        # The module-level geodetic_agent also calls build_llm at import time;
        # we only care that the last call used None.
        calls = mock_build.call_args_list
        assert any(c == call(None) for c in calls)

    def test_create_geodetic_agent_returns_create_agent_result(self):
        """The factory returns whatever create_agent() returns."""
        sentinel = MagicMock(name="agent_sentinel")
        mock_build = MagicMock(return_value=MagicMock())
        mock_create = MagicMock(return_value=sentinel)

        with patch("src.agents.llm_factory.build_llm", mock_build), \
             patch("langchain.agents.create_agent", mock_create):
            from src.agents import geodetic
            importlib.reload(geodetic)
            result = geodetic.create_geodetic_agent()

        assert result is sentinel

    def test_create_geodetic_agent_passes_tools(self):
        """create_agent is called with the canonical TOOLS list."""
        mock_build = MagicMock(return_value=MagicMock())
        mock_create = MagicMock(return_value=MagicMock())

        with patch("src.agents.llm_factory.build_llm", mock_build), \
             patch("langchain.agents.create_agent", mock_create):
            from src.agents import geodetic
            importlib.reload(geodetic)
            geodetic.create_geodetic_agent()

        # get the last call (the explicit factory call, not the module-level one)
        call_kwargs = mock_create.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) > 0

    def test_module_exposes_system_prompt_constant(self):
        """The module must expose a non-empty SYSTEM_PROMPT string constant."""
        mock_build = MagicMock(return_value=MagicMock())
        mock_create = MagicMock(return_value=MagicMock())

        with patch("src.agents.llm_factory.build_llm", mock_build), \
             patch("langchain.agents.create_agent", mock_create):
            from src.agents import geodetic
            importlib.reload(geodetic)

        assert hasattr(geodetic, "SYSTEM_PROMPT")
        assert isinstance(geodetic.SYSTEM_PROMPT, str)
        assert len(geodetic.SYSTEM_PROMPT) > 100  # substantive content

    def test_module_exposes_tools_constant(self):
        """The module must expose a non-empty TOOLS list constant."""
        mock_build = MagicMock(return_value=MagicMock())
        mock_create = MagicMock(return_value=MagicMock())

        with patch("src.agents.llm_factory.build_llm", mock_build), \
             patch("langchain.agents.create_agent", mock_create):
            from src.agents import geodetic
            importlib.reload(geodetic)

        assert hasattr(geodetic, "TOOLS")
        assert isinstance(geodetic.TOOLS, list)
        assert len(geodetic.TOOLS) >= 4

    def test_module_level_geodetic_agent_exists(self):
        """The module must expose a module-level geodetic_agent instance."""
        mock_build = MagicMock(return_value=MagicMock())
        mock_create = MagicMock(return_value=MagicMock())

        with patch("src.agents.llm_factory.build_llm", mock_build), \
             patch("langchain.agents.create_agent", mock_create):
            from src.agents import geodetic
            importlib.reload(geodetic)

        assert hasattr(geodetic, "geodetic_agent")
        assert geodetic.geodetic_agent is not None
