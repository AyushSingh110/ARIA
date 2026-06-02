from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

# Sandboxed to a workspace directory so the agent cannot write outside it
_WORKSPACE = Path("workspace")


def _safe_path(relative_path: str) -> Path:
    _WORKSPACE.mkdir(parents=True, exist_ok=True)
    target = (_WORKSPACE / relative_path).resolve()
    workspace_resolved = _WORKSPACE.resolve()
    # Prevent path traversal outside workspace
    if not str(target).startswith(str(workspace_resolved)):
        raise ValueError(f"Path '{relative_path}' escapes the workspace sandbox.")
    return target


@tool
def write_file(path: str, content: str) -> str:
    """Write text content to a file inside the workspace directory.

    Args:
        path: Relative file path, e.g. "output/report.txt"
        content: Text content to write.

    Returns:
        Confirmation message with the resolved path, or an error.
    """
    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to workspace/{path}"
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error writing file: {exc}"


@tool
def read_file(path: str) -> str:
    """Read text content from a file inside the workspace directory.

    Args:
        path: Relative file path, e.g. "output/report.txt"

    Returns:
        File content as a string, or an error message if not found.
    """
    try:
        target = _safe_path(path)
        if not target.exists():
            return f"Error: file 'workspace/{path}' does not exist."
        return target.read_text(encoding="utf-8")
    except ValueError as exc:
        return f"Error: {exc}"
    except OSError as exc:
        return f"Error reading file: {exc}"
