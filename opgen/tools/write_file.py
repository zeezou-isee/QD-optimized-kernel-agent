"""
WriteFile tool - Creates or overwrites a file on the local filesystem.

Inspired by Claude Code's FileWriteTool. Creates parent directories automatically.
Used for creating new files or doing complete rewrites of existing files.
"""

import os
from typing import Any


TOOL_NAME = "write_file"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Write a file to the local filesystem. Creates parent directories if needed. Overwrites existing files. Prefer the edit_file tool for partial modifications.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative)."
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file."
                }
            },
            "required": ["file_path", "content"]
        }
    }
}


def write_file(file_path: str, content: str) -> dict[str, Any]:
    """
    Write content to a file, creating it or overwriting it.

    Args:
        file_path: Absolute path to the file to write.
        content: String content to write.

    Returns:
        A dict with success status and metadata about the operation.
    """
    if not os.path.isabs(file_path):
        return {
            "success": False,
            "error": f"file_path must be an absolute path, got: {file_path}"
        }

    existed = os.path.exists(file_path)
    old_content = None

    if existed:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                old_content = f.read()
        except Exception:
            pass  # Binary or unreadable file — treat as overwrite

    # Ensure parent directory exists
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception as e:
            return {"success": False, "error": f"Failed to create parent directory: {e}"}

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return {"success": False, "error": f"Failed to write file: {e}"}

    result = {
        "success": True,
        "type": "update" if existed else "create",
        "file_path": file_path,
        "content": content,
        "original_content": old_content
    }

    return result
