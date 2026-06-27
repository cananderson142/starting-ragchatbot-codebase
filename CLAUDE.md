# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.13+, uv required)
uv sync

# Run the app (from repo root). Starts uvicorn on :8000 with --reload.
./run.sh
# Equivalent:
cd backend && uv run uvicorn app:app --reload --port 8000

# Force a full re-index of ./docs on next start
rm -rf backend/chroma_db

# One-off Python against the project env
cd backend && uv run python -c "..."
```

`.env` (repo root) must contain `ANTHROPIC_API_KEY=...`. There is no test suite, linter, or formatter configured.

**Always use `uv` for all dependency and env operations.** Deps live in `pyproject.toml` + `uv.lock`; anything that bypasses uv desyncs the lockfile. Use `uv sync` to install, `uv add <pkg>` / `uv remove <pkg>` to manage deps, `uv run <cmd>` to execute anything in the project env. Never use `pip`, bare `python`, or manual venv activation.

## Architecture

This is a tool-augmented RAG system: **Claude decides when to retrieve**, rather than the backend retrieving on every query. Understanding this inversion is essential.

### Query path
`POST /api/query` (`backend/app.py:56`) → `RAGSystem.query` (`backend/rag_system.py:102`) → `AIGenerator.generate_response` (`backend/ai_generator.py:43`) makes a **first** call to Claude with the `search_course_content` tool exposed. If Claude returns `stop_reason == "tool_use"`, `_handle_tool_execution` (`ai_generator.py:89`) runs the tool, appends the result, and makes a **second** call (without tools) to compose the final answer. If Claude doesn't invoke the tool, the first response is returned directly.

The system prompt in `ai_generator.py:8` enforces **one search per query maximum**, and the code only handles a single tool→continuation cycle (no multi-step research loop). Adding multi-hop retrieval requires changing both.

### Two ChromaDB collections (`backend/vector_store.py`)
- `course_catalog` — one row per course; ID = course title; metadata includes `lessons_json`. Used by `_resolve_course_name` (`vector_store.py:102`) for fuzzy course-name matching when the tool is called with a `course_name` argument.
- `course_content` — one row per chunk; ID = `<title_with_underscores>_<chunk_index>`. The actual retrieval target.

Both use `all-MiniLM-L6-v2` embeddings. `_resolve_course_name` returns the top-1 catalog hit **with no distance threshold** — a typo or unknown course name silently resolves to the least-different course rather than returning empty. Worth knowing before debugging unexpected results.

### Source tracking (cross-component coupling)
`CourseSearchTool` writes hit labels to `self.last_sources` (`search_tools.py:88`). `RAGSystem.query` reads them via `ToolManager.get_last_sources()` then calls `reset_sources()` (`rag_system.py:130-133`). This is per-tool-instance mutable state, so concurrent `/api/query` calls can race. New tools that should surface sources to the UI must expose a `last_sources` attribute — `ToolManager` discovers it by `hasattr`.

### Session state
`SessionManager` (`backend/session_manager.py`) is an in-process dict, wiped on restart. `MAX_HISTORY=2` exchanges (4 messages) are retained and injected as a formatted string into the system prompt — *not* as proper message turns.

### Document ingestion
Triggered once at startup (`app.py:88`) over `./docs`. Idempotent by course title: existing titles are skipped (`rag_system.py:87`), so editing a doc requires deleting `backend/chroma_db/`.

`DocumentProcessor.process_course_document` (`backend/document_processor.py:97`) expects this exact header format:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <lesson title>
Lesson Link: <url>
<lesson body...>
Lesson 1: ...
```

`chunk_text` (`document_processor.py:25`) is sentence-aware with `CHUNK_SIZE=800` / `CHUNK_OVERLAP=100` (configurable in `config.py`). Two near-duplicate code paths handle "current lesson" (`:185`) vs. "final lesson" (`:234`) with slightly different context prefixes — preserve both when refactoring. `.pdf` and `.docx` are listed as accepted extensions but `read_file` only does plain text decoding, so they'll fail.

### Config
All tunables live in `backend/config.py` as a single `Config` dataclass: model ID (`claude-sonnet-4-20250514` is hardcoded), embedding model, chunk size/overlap, `MAX_RESULTS`, `MAX_HISTORY`, ChromaDB path. The `config` singleton is imported directly; there is no environment override beyond `ANTHROPIC_API_KEY`.

### Frontend
Plain `frontend/index.html` + `script.js` + `style.css`, no build step. Served as static files mounted at `/` by FastAPI (`app.py:119`) with no-cache headers (`DevStaticFiles` at `app.py:107`). It calls only `POST /api/query` and `GET /api/courses`.
