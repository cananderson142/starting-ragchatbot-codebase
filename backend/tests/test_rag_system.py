"""Tests for RAGSystem.query against a real ChromaDB but a mocked Anthropic client.

We verify the orchestration the RAG layer is responsible for:
- end-to-end content query path when Claude invokes the search tool
- no-tool path when Claude answers directly
- sources are returned from the tool then reset
- session history is persisted and re-injected on subsequent turns
"""

from types import SimpleNamespace
from unittest.mock import MagicMock


def _text_response(text):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
    )


def _tool_use_response(tool_input, tool_id="tu_1"):
    block = SimpleNamespace(
        type="tool_use",
        name="search_course_content",
        input=tool_input,
        id=tool_id,
    )
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def test_content_query_invokes_search_and_returns_sources(rag):
    """Simulates Claude using the search tool; verifies the full plumbing."""
    rag.ai_generator.client = MagicMock()
    rag.ai_generator.client.messages.create.side_effect = [
        _tool_use_response({"query": "MCP"}),
        _text_response("Synthesized answer about MCP."),
    ]

    answer, sources = rag.query("what is MCP?")

    assert answer == "Synthesized answer about MCP."
    assert len(sources) > 0, "expected the search tool to surface sources"
    assert all(isinstance(s, str) for s in sources)


def test_general_query_no_tool_use(rag):
    rag.ai_generator.client = MagicMock()
    rag.ai_generator.client.messages.create.return_value = _text_response("4")

    answer, sources = rag.query("what is 2+2?")
    assert answer == "4"
    assert sources == []


def test_sources_reset_after_query(rag):
    rag.ai_generator.client = MagicMock()
    rag.ai_generator.client.messages.create.side_effect = [
        _tool_use_response({"query": "MCP"}),
        _text_response("answer"),
    ]

    rag.query("q")
    assert rag.search_tool.last_sources == []


def test_session_persists_exchange(rag):
    rag.ai_generator.client = MagicMock()
    rag.ai_generator.client.messages.create.return_value = _text_response("hi")

    sid = rag.session_manager.create_session()
    rag.query("first question", session_id=sid)

    history = rag.session_manager.get_conversation_history(sid)
    assert "first question" in history
    assert "hi" in history


def test_prior_history_injected_into_system_prompt(rag):
    rag.ai_generator.client = MagicMock()
    rag.ai_generator.client.messages.create.return_value = _text_response("turn2")

    sid = rag.session_manager.create_session()
    rag.session_manager.add_exchange(sid, "prior q", "prior a")
    rag.query("new q", session_id=sid)

    system_arg = rag.ai_generator.client.messages.create.call_args.kwargs["system"]
    assert "prior q" in system_arg
    assert "prior a" in system_arg


def test_search_tool_receives_arguments_from_claude(rag):
    """When Claude tool-calls with course_name + lesson_number, the tool sees them."""
    rag.ai_generator.client = MagicMock()
    rag.ai_generator.client.messages.create.side_effect = [
        _tool_use_response({"query": "tools", "course_name": "MCP", "lesson_number": 2}),
        _text_response("ok"),
    ]

    _, sources = rag.query("what tools does MCP support in lesson 2?")
    assert sources, "expected filtered sources from MCP Lesson 2"
    assert all("Lesson 2" in s for s in sources)
