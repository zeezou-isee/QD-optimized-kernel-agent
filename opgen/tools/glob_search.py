"""
GlobSearch tool - Find files by glob pattern matching.

Inspired by Claude Code's GlobTool. Returns matching file paths sorted by
modification time. Useful for discovering files by naming patterns.
"""

import os
import glob as glob_module
from typing import Any, Optional


TOOL_NAME = "glob_search"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Find files matching a glob pattern. Returns file paths sorted by modification time (newest first). Use for discovering files by naming patterns.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match (e.g. '**/*.cpp', 'src/**/*.h', '*.py')."
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in. Defaults to current working directory."
                }
            },
            "required": ["pattern"]
        }
    }
}

MAX_RESULTS = 100


def glob_search(pattern: str, path: Optional[str] = None) -> dict[str, Any]:
    """
    Find files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., '**/*.cpp', 'src/**/*.h').
        path: Root directory to search. Defaults to CWD.

    Returns:
        A dict with matching file paths sorted by mtime.
    """
    search_root = path or os.getcwd()
    if not os.path.isabs(search_root):
        search_root = os.path.abspath(search_root)

    if not os.path.isdir(search_root):
        return {"success": False, "error": f"Directory does not exist: {search_root}"}

    # Use recursive glob if pattern starts with **
    full_pattern = os.path.join(search_root, pattern)
    results = glob_module.glob(full_pattern, recursive=True, include_hidden=False)

    # Filter only files (not directories)
    files = [f for f in results if os.path.isfile(f)]

    # Sort by modification time (newest first)
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)

    truncated = len(files) > MAX_RESULTS
    if truncated:
        files = files[:MAX_RESULTS]

    return {
        "success": True,
        "pattern": pattern,
        "search_path": search_root,
        "num_files": len(files),
        "filenames": files,
        "truncated": truncated
    }
