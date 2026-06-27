"""Tests for CourseSearchTool.execute() against the real ChromaDB."""


def test_execute_plain_query_returns_formatted_results(search_tool):
    out = search_tool.execute(query="how does MCP work?")
    assert out.strip(), "expected non-empty results text"
    assert "[" in out and "]" in out, "expected '[Course - Lesson N]' context headers"


def test_execute_populates_last_sources(search_tool):
    assert search_tool.last_sources == []
    search_tool.execute(query="how does MCP work?")
    assert len(search_tool.last_sources) > 0
    assert all(isinstance(s, str) for s in search_tool.last_sources)


def test_execute_with_course_filter_restricts_to_that_course(search_tool):
    search_tool.execute(query="embeddings", course_name="Chroma")
    assert search_tool.last_sources, "expected sources from the Chroma course"
    assert all("Chroma" in s for s in search_tool.last_sources), search_tool.last_sources


def test_execute_with_course_and_lesson_filter(search_tool):
    search_tool.execute(query="tools", course_name="MCP", lesson_number=2)
    assert search_tool.last_sources, "expected sources from MCP lesson 2"
    assert all("Lesson 2" in s for s in search_tool.last_sources), search_tool.last_sources


def test_execute_with_strict_filter_no_match(search_tool):
    out = search_tool.execute(query="anything", course_name="MCP", lesson_number=999)
    assert "No relevant content found" in out


def test_execute_unknown_course_should_return_no_match(search_tool):
    """A nonsense course name should not silently resolve to the nearest course.
    Currently fails: `_resolve_course_name` has no distance threshold."""
    out = search_tool.execute(query="anything", course_name="NonexistentCourseXYZ")
    assert "No course found matching" in out, (
        f"Expected 'No course found matching ...' for unknown course, got:\n"
        f"  output (truncated): {out[:200]!r}\n"
        f"  last_sources: {search_tool.last_sources}"
    )


def test_get_tool_definition_schema(search_tool):
    d = search_tool.get_tool_definition()
    assert d["name"] == "search_course_content"
    assert "query" in d["input_schema"]["required"]
    assert set(d["input_schema"]["properties"]) == {
        "query",
        "course_name",
        "lesson_number",
    }
