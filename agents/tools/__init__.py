"""
Operator Agent Tools - Unified interface for LLM function calling.

This module provides all tools needed by an LLM agent to write ncnn operators:
  - read_file    : Read files from the filesystem
  - write_file   : Create or overwrite files
  - edit_file    : Perform exact string replacements in files
  - grep_search  : Search file contents with regex
  - glob_search  : Find files by glob pattern
  - bash_exec    : Execute shell commands

Usage:
    from operator_agent.tools import TOOL_SCHEMAS, execute_tool

    # Get all tool schemas for LLM function calling
    schemas = TOOL_SCHEMAS

    # Execute a tool call from the LLM
    result = execute_tool("read_file", file_path="/path/to/file.cpp")
"""

from typing import Any, Callable

from .read_file import SCHEMA as READ_FILE_SCHEMA, read_file, TOOL_NAME as READ_FILE_NAME
from .write_file import SCHEMA as WRITE_FILE_SCHEMA, write_file, TOOL_NAME as WRITE_FILE_NAME
from .edit_file import SCHEMA as EDIT_FILE_SCHEMA, edit_file, TOOL_NAME as EDIT_FILE_NAME
from .grep_search import SCHEMA as GREP_SEARCH_SCHEMA, grep_search, TOOL_NAME as GREP_SEARCH_NAME
from .glob_search import SCHEMA as GLOB_SEARCH_SCHEMA, glob_search, TOOL_NAME as GLOB_SEARCH_NAME
from .bash_exec import SCHEMA as BASH_EXEC_SCHEMA, bash_exec, TOOL_NAME as BASH_EXEC_NAME

__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
    "TOOL_MAP",
    "execute_tool",
    "get_tool_schema",
    "read_file",
    "write_file",
    "edit_file",
    "grep_search",
    "glob_search",
    "bash_exec",
]

# All tool schemas in OpenAI function-calling format
TOOL_SCHEMAS: list[dict[str, Any]] = [
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    EDIT_FILE_SCHEMA,
    GREP_SEARCH_SCHEMA,
    GLOB_SEARCH_SCHEMA,
    BASH_EXEC_SCHEMA,
]

# Map tool name -> (function, schema) for dispatch
TOOL_MAP: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
    READ_FILE_NAME: (read_file, READ_FILE_SCHEMA),
    WRITE_FILE_NAME: (write_file, WRITE_FILE_SCHEMA),
    EDIT_FILE_NAME: (edit_file, EDIT_FILE_SCHEMA),
    GREP_SEARCH_NAME: (grep_search, GREP_SEARCH_SCHEMA),
    GLOB_SEARCH_NAME: (glob_search, GLOB_SEARCH_SCHEMA),
    BASH_EXEC_NAME: (bash_exec, BASH_EXEC_SCHEMA),
}

# Convenience: just the callables keyed by name
TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    name: func for name, (func, _) in TOOL_MAP.items()
}


def execute_tool(name: str, **kwargs) -> dict[str, Any]:
    """
    Execute a tool by name with the given keyword arguments.

    This is the primary entry point for LLM tool dispatch.

    Args:
        name: Tool name (e.g., "read_file", "grep_search").
        **kwargs: Tool-specific parameters.

    Returns:
        A dict with at least a 'success' key.

    Raises:
        ValueError: If the tool name is not recognized.

    Example:
        >>> result = execute_tool("read_file", file_path="/tmp/foo.cpp")
        >>> print(result["success"])
        True
    """
    entry = TOOL_MAP.get(name)
    if entry is None:
        return {
            "success": False,
            "error": f"Unknown tool: '{name}'. Available tools: {list(TOOL_MAP.keys())}"
        }
    func, _ = entry
    try:
        return func(**kwargs)
    except TypeError as e:
        return {"success": False, "error": f"Invalid arguments for tool '{name}': {e}"}
    except Exception as e:
        return {"success": False, "error": f"Tool '{name}' raised an exception: {e}"}


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """
    Get the JSON schema for a specific tool.

    Args:
        name: Tool name.

    Returns:
        The tool's function-calling schema dict, or None if not found.
    """
    entry = TOOL_MAP.get(name)
    return entry[1] if entry else None
