"""
EditFile tool - Performs exact string replacements in existing files.

Inspired by Claude Code's FileEditTool. Uses old_string/new_string replacement.
Supports replace_all for global find-and-replace.
"""

import os
from typing import Any


TOOL_NAME = "edit_file"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Perform exact string replacements in an existing file. The edit will FAIL if old_string is not unique in the file. Use replace_all to change every occurrence of old_string.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify."
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to replace in the file."
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences of old_string. Defaults to false.",
                    "default": False
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        }
    }
}


def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False
) -> dict[str, Any]:
    """
    Edit a file by replacing exact string matches.

    Args:
        file_path: Absolute path to the file to edit.
        old_string: The exact text to find and replace.
        new_string: The replacement text.
        replace_all: If True, replace all occurrences. If False, fail if more than one match.

    Returns:
        A dict with success status, patch info, and result metadata.
    """
    if not os.path.isabs(file_path):
        return {
            "success": False,
            "error": f"file_path must be an absolute path, got: {file_path}"
        }

    if old_string == new_string:
        return {
            "success": False,
            "error": "old_string and new_string are identical — no changes to make."
        }

    if not os.path.exists(file_path):
        if old_string == "":
            # Create new file
            return write_file_inline(file_path, new_string)
        return {
            "success": False,
            "error": f"File does not exist: {file_path}. To create a new file, use write_file or pass old_string=''."
        }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original = f.read()
    except Exception as e:
        return {"success": False, "error": f"Failed to read file: {e}"}

    count = original.count(old_string)
    if count == 0:
        return {
            "success": False,
            "error": f"old_string not found in file. String: {old_string[:200]}"
        }

    if count > 1 and not replace_all:
        return {
            "success": False,
            "error": (
                f"Found {count} occurrences of old_string, but replace_all is False. "
                f"Use replace_all=True to replace all, or provide more context to uniquely identify one instance."
            )
        }

    new_content = original.replace(old_string, new_string)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as e:
        return {"success": False, "error": f"Failed to write file: {e}"}

    return {
        "success": True,
        "file_path": file_path,
        "occurrences_replaced": count if replace_all else 1,
        "replace_all": replace_all,
        "old_string": old_string,
        "new_string": new_string,
        "original_content": original,
        "new_content": new_content
    }


def write_file_inline(file_path: str, content: str) -> dict[str, Any]:
    """Internal helper: create a new file via edit when old_string is empty."""
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
        return {"success": False, "error": f"Failed to write new file: {e}"}

    return {
        "success": True,
        "file_path": file_path,
        "type": "create",
        "new_content": content
    }
