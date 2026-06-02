"""Unit tests for Phase 1 tools — no LLM calls, no API keys required."""
from __future__ import annotations

import pytest


# ── Calculator ────────────────────────────────────────────────────────────────

from aria.tools.calculator import calculator


def test_calculator_basic_arithmetic():
    assert calculator.invoke({"expression": "2 + 3"}) == "5"
    assert calculator.invoke({"expression": "10 - 4"}) == "6"
    assert calculator.invoke({"expression": "6 * 7"}) == "42"
    assert calculator.invoke({"expression": "10 / 4"}) == "2.5"


def test_calculator_power_and_mod():
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"
    assert calculator.invoke({"expression": "17 % 5"}) == "2"
    assert calculator.invoke({"expression": "17 // 5"}) == "3"


def test_calculator_math_functions():
    result = calculator.invoke({"expression": "sqrt(144)"})
    assert result == "12.0"

    result = calculator.invoke({"expression": "round(3.14159, 2)"})
    assert result == "3.14"


def test_calculator_constants():
    result = calculator.invoke({"expression": "pi"})
    assert result.startswith("3.14")


def test_calculator_division_by_zero():
    result = calculator.invoke({"expression": "1 / 0"})
    assert "division by zero" in result.lower()


def test_calculator_rejects_unsafe_code():
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert "Error" in result


def test_calculator_invalid_syntax():
    result = calculator.invoke({"expression": "2 +"})
    assert "Error" in result


# ── File ops ──────────────────────────────────────────────────────────────────

from aria.tools.file_ops import read_file, write_file


def test_write_and_read_file(tmp_path, monkeypatch):
    # Redirect workspace to tmp_path
    import aria.tools.file_ops as fo
    monkeypatch.setattr(fo, "_WORKSPACE", tmp_path / "workspace")

    result = write_file.invoke({"path": "test.txt", "content": "hello aria"})
    assert "Written" in result

    content = read_file.invoke({"path": "test.txt"})
    assert content == "hello aria"


def test_read_nonexistent_file(tmp_path, monkeypatch):
    import aria.tools.file_ops as fo
    monkeypatch.setattr(fo, "_WORKSPACE", tmp_path / "workspace")

    result = read_file.invoke({"path": "ghost.txt"})
    assert "does not exist" in result


def test_write_file_path_traversal(tmp_path, monkeypatch):
    import aria.tools.file_ops as fo
    monkeypatch.setattr(fo, "_WORKSPACE", tmp_path / "workspace")

    result = write_file.invoke({"path": "../../etc/passwd", "content": "bad"})
    assert "Error" in result


# ── Web search ────────────────────────────────────────────────────────────────

from aria.tools.web_search import web_search


def test_web_search_returns_string():
    result = web_search.invoke({"query": "Python programming language"})
    assert isinstance(result, str)
    assert len(result) > 10


def test_web_search_empty_query():
    result = web_search.invoke({"query": ""})
    assert "Error" in result


def test_web_search_known_keyword():
    result = web_search.invoke({"query": "langgraph multi-agent"})
    assert "LangGraph" in result or "langgraph" in result.lower()
