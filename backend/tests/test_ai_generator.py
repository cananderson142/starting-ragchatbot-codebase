"""Tests for AIGenerator with a mocked Anthropic client.

These tests verify that AIGenerator correctly:
- returns text directly when Claude doesn't call a tool
- triggers tool execution and a second API call when Claude returns tool_use
- forwards tool inputs verbatim and assembles the follow-up messages correctly
- injects conversation history into the system prompt
- passes the configured model / temperature / max_tokens / tools to the API
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator


def _text_response(text="hello"):
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
    )


def _tool_use_response(tool_input=None, tool_id="tu_1", name="search_course_content"):
    block = SimpleNamespace(
        type="tool_use",
        name=name,
        input=tool_input or {"query": "MCP"},
        id=tool_id,
    )
    return SimpleNamespace(stop_reason="tool_use", content=[block])


@pytest.fixture
def gen():
    g = AIGenerator(api_key="test-key", model="claude-test-model")
    g.client = MagicMock()
    return g


def test_no_tool_use_returns_text_directly(gen):
    gen.client.messages.create.return_value = _text_response("direct answer")
    result = gen.generate_response(query="hi", tools=None, tool_manager=None)
    assert result == "direct answer"
    assert gen.client.messages.create.call_count == 1


def test_tool_use_executes_tool_and_makes_second_call(gen):
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"
    gen.client.messages.create.side_effect = [
        _tool_use_response(tool_input={"query": "MCP basics", "lesson_number": 2}),
        _text_response("final answer"),
    ]

    result = gen.generate_response(
        query="what is MCP?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "final answer"
    assert gen.client.messages.create.call_count == 2
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="MCP basics", lesson_number=2
    )


def test_second_call_omits_tools(gen):
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "x"
    gen.client.messages.create.side_effect = [
        _tool_use_response(),
        _text_response("done"),
    ]

    gen.generate_response(
        query="q",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_kwargs = gen.client.messages.create.call_args_list[1].kwargs
    assert "tools" not in second_kwargs
    assert "tool_choice" not in second_kwargs


def test_message_sequence_after_tool_use(gen):
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool result text"
    gen.client.messages.create.side_effect = [
        _tool_use_response(tool_id="tu_42"),
        _text_response("final"),
    ]

    gen.generate_response(
        query="q",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    msgs = gen.client.messages.create.call_args_list[1].kwargs["messages"]
    assert msgs[0] == {"role": "user", "content": "q"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    block = msgs[2]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu_42"
    assert block["content"] == "tool result text"


def test_history_injected_into_system_prompt(gen):
    gen.client.messages.create.return_value = _text_response("ok")
    gen.generate_response(query="q", conversation_history="User: hi\nAssistant: hello")
    system_arg = gen.client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:" in system_arg
    assert "User: hi" in system_arg


def test_api_call_uses_configured_model_and_params(gen):
    gen.client.messages.create.return_value = _text_response("ok")
    gen.generate_response(query="q")
    kwargs = gen.client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-test-model"
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 800


def test_tools_param_added_with_auto_choice(gen):
    gen.client.messages.create.return_value = _text_response("ok")
    tools = [{"name": "search_course_content"}]
    gen.generate_response(query="q", tools=tools)
    kwargs = gen.client.messages.create.call_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == {"type": "auto"}
