"""End-to-end smoke test against the real Anthropic API.

This catches model deprecation / API errors that mocked tests can never see.
Set `SKIP_LIVE=1` to skip when offline.
"""

import os

import pytest


@pytest.mark.skipif(os.environ.get("SKIP_LIVE") == "1", reason="SKIP_LIVE=1 set")
def test_content_query_against_real_anthropic_api(rag):
    answer, sources = rag.query("what is MCP?")
    assert isinstance(answer, str) and answer.strip(), (
        f"empty answer; sources={sources}"
    )
