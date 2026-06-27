import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import pytest

from config import config as _config
from rag_system import RAGSystem
from search_tools import CourseSearchTool
from vector_store import VectorStore


@pytest.fixture(scope="session")
def chroma_path():
    p = BACKEND_DIR / "chroma_db"
    if not p.exists():
        pytest.skip("chroma_db not present — start the app once to ingest /docs")
    return str(p)


@pytest.fixture(scope="session")
def store(chroma_path):
    return VectorStore(chroma_path, _config.EMBEDDING_MODEL, max_results=5)


@pytest.fixture
def search_tool(store):
    return CourseSearchTool(store)


@pytest.fixture
def rag(chroma_path):
    """RAGSystem with CHROMA_PATH pointed at backend/chroma_db absolutely.

    The app runs from cwd=backend/ in production (run.sh does `cd backend`),
    so config.CHROMA_PATH='./chroma_db' works there but not under pytest.
    """
    original = _config.CHROMA_PATH
    _config.CHROMA_PATH = chroma_path
    try:
        yield RAGSystem(_config)
    finally:
        _config.CHROMA_PATH = original
