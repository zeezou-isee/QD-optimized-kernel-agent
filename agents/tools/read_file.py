"""
ReadFile tool - Reads a file from the local filesystem.

Inspired by Claude Code's FileReadTool. Supports reading text files with
optional offset/limit for large files, plus image and PDF files.
"""

import os
import base64
from typing import Any, Optional


TOOL_NAME = "read_file"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Read a file from the local filesystem. Supports text files, images, and PDFs. Returns file content with line numbers for text files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read."
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from. Only provide if the file is too large to read at once."
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once."
                }
            },
            "required": ["file_path"]
        }
    }
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_LINES_DEFAULT = 2000


def read_file(file_path: str, offset: int = 1, limit: Optional[int] = None) -> dict[str, Any]:
    """
    Read a file from the local filesystem.

    Args:
        file_path: Absolute path to the file.
        offset: Line number to start reading from (1-indexed). Defaults to 1.
        limit: Max number of lines to read. Defaults to 2000.

    Returns:
        A dict with type, content, and metadata.
    """
    if not os.path.isabs(file_path):
        return {
            "success": False,
            "error": f"file_path must be an absolute path, got: {file_path}"
        }

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File does not exist: {file_path}"
        }

    ext = os.path.splitext(file_path)[1].lower()

    # Image handling
    if ext in IMAGE_EXTENSIONS:
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("ascii")
            mime_map = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"
            }
            return {
                "success": True,
                "type": "image",
                "file_path": file_path,
                "mime_type": mime_map.get(ext, "application/octet-stream"),
                "base64": b64,
                "size_bytes": len(data)
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to read image: {e}"}

    # Text file handling
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        except Exception as e:
            return {"success": False, "error": f"Cannot read file as text: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {e}"}

    total_lines = len(lines)
    start_idx = max(0, offset - 1)
    end_idx = start_idx + (limit or MAX_LINES_DEFAULT)
    selected = lines[start_idx:end_idx]

    # Format with line numbers
    content_with_numbers = ""
    for i, line in enumerate(selected):
        content_with_numbers += f"{start_idx + i + 1}\t{line}"

    result = {
        "success": True,
        "type": "text",
        "file_path": file_path,
        "content": content_with_numbers.rstrip("\n"),
        "num_lines": len(selected),
        "start_line": offset,
        "total_lines": total_lines
    }

    # Warn if truncated
    if end_idx < total_lines:
        result["truncated"] = True
        result["truncated_note"] = (
            f"File has {total_lines} total lines. "
            f"Showing lines {offset}-{start_idx + len(selected)}. "
            f"Use offset={start_idx + len(selected) + 1} to read more."
        )

    return result
