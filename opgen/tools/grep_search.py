"""
GrepSearch tool - Search file contents using regex patterns.

Inspired by Claude Code's GrepTool (which wraps ripgrep).
Supports full regex syntax, file type filtering, glob filtering,
content/files_with_matches/count output modes, and context lines.
"""

import os
import re
import fnmatch
from typing import Any, Optional


TOOL_NAME = "grep_search"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Search file contents using regex patterns. Supports glob filtering, file type filtering, context lines, and multiple output modes. Use this for all code search tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents."
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in. Defaults to current working directory."
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. '*.py', '*.{cpp,h}')."
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode: 'content' shows matching lines, 'files_with_matches' shows file paths (default), 'count' shows match counts per file."
                },
                "-A": {
                    "type": "integer",
                    "description": "Number of lines to show after each match (context after)."
                },
                "-B": {
                    "type": "integer",
                    "description": "Number of lines to show before each match (context before)."
                },
                "-C": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match."
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search. Default: false."
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Limit output to first N lines/entries. Defaults to 250."
                }
            },
            "required": ["pattern"]
        }
    }
}

# Directories to always exclude from search
EXCLUDE_DIRS = {'.git', '.svn', '.hg', '__pycache__', 'node_modules', '.claude', 'build', 'dist'}

DEFAULT_HEAD_LIMIT = 250

# File type to extension mapping (subset)
TYPE_EXTENSIONS = {
    "py": {".py"},
    "js": {".js", ".jsx", ".mjs", ".cjs"},
    "ts": {".ts", ".tsx"},
    "cpp": {".cpp", ".cc", ".cxx", ".c++"},
    "c": {".c"},
    "h": {".h", ".hpp", ".hxx"},
    "rust": {".rs"},
    "go": {".go"},
    "java": {".java"},
    "pyx": {".pyx", ".pxd", ".pxi"},
}


def grep_search(
    pattern: str,
    path: Optional[str] = None,
    glob: Optional[str] = None,
    output_mode: str = "files_with_matches",
    **kwargs
) -> dict[str, Any]:
    """
    Search files for lines matching a regex pattern.

    Args:
        pattern: Regex pattern to search for.
        path: Directory or file to search. Defaults to CWD.
        glob: Glob pattern to filter files (e.g., '*.py', '*.{cpp,h}').
        output_mode: 'content', 'files_with_matches', or 'count'.
        **kwargs: -A, -B, -C (context lines), -i (case insensitive), head_limit.

    Returns:
        A dict with search results.
    """
    search_root = path or os.getcwd()
    if not os.path.isabs(search_root):
        search_root = os.path.abspath(search_root)

    if not os.path.exists(search_root):
        return {"success": False, "error": f"Path does not exist: {search_root}"}

    context_before = kwargs.get("-B", 0) or 0
    context_after = kwargs.get("-A", 0) or 0
    context_around = kwargs.get("-C", 0) or 0
    if context_around:
        context_before = context_after = context_around

    case_insensitive = kwargs.get("-i", False)
    head_limit = kwargs.get("head_limit", DEFAULT_HEAD_LIMIT)
    if head_limit == 0:
        head_limit = None  # Unlimited

    # Compile regex
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return {"success": False, "error": f"Invalid regex pattern: {e}"}

    # Collect files to search
    files = _collect_files(search_root, glob)

    # Search
    if output_mode == "content":
        return _search_content(files, regex, context_before, context_after, head_limit)
    elif output_mode == "count":
        return _search_count(files, regex, head_limit)
    else:
        return _search_files_with_matches(files, regex, head_limit)


def _collect_files(root: str, glob_pattern: Optional[str]) -> list[str]:
    """Collect all text files under root, optionally filtered by glob."""
    files = []
    if os.path.isfile(root):
        files.append(root)
        return files

    for dirpath, dirnames, filenames in os.walk(root):
        # Exclude hidden/VCS directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith('.')]

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if glob_pattern:
                if not fnmatch.fnmatch(fname, glob_pattern):
                    # Also try matching against full relative path
                    rel = os.path.relpath(fpath, root)
                    if not fnmatch.fnmatch(rel, glob_pattern):
                        continue
            # Skip binary-looking files
            if _is_likely_binary(fpath):
                continue
            files.append(fpath)

    return files


def _is_likely_binary(filepath: str) -> bool:
    """Quick check if a file is likely binary."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read(1024)
        return False
    except UnicodeDecodeError:
        return True
    except Exception:
        return True


def _search_files_with_matches(
    files: list[str], regex: re.Pattern, head_limit: Optional[int]
) -> dict[str, Any]:
    """Search for files containing the pattern."""
    matched = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                if regex.search(f.read()):
                    matched.append(fpath)
                    if head_limit and len(matched) >= head_limit:
                        break
        except Exception:
            pass

    return {
        "success": True,
        "mode": "files_with_matches",
        "num_files": len(matched),
        "filenames": matched,
        "truncated": head_limit is not None and len(matched) >= head_limit
    }


def _search_count(
    files: list[str], regex: re.Pattern, head_limit: Optional[int]
) -> dict[str, Any]:
    """Search and count matches per file."""
    counts = {}
    total = 0
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            c = len(regex.findall(content))
            if c > 0:
                counts[fpath] = c
                total += c
                if head_limit and len(counts) >= head_limit:
                    break
        except Exception:
            pass

    return {
        "success": True,
        "mode": "count",
        "num_files": len(counts),
        "filenames": list(counts.keys()),
        "counts": counts,
        "total_matches": total,
        "truncated": head_limit is not None and len(counts) >= head_limit
    }


def _search_content(
    files: list[str],
    regex: re.Pattern,
    context_before: int,
    context_after: int,
    head_limit: Optional[int]
) -> dict[str, Any]:
    """Search and return matching lines with context."""
    results = []
    line_count = 0

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            m = regex.search(line)
            if not m:
                continue

            if head_limit is not None and line_count >= head_limit:
                break

            # Get context
            start = max(0, i - context_before)
            end = min(len(lines), i + context_after + 1)

            if context_before > 0 or context_after > 0:
                for j in range(start, end):
                    prefix = ">" if j == i else " "
                    rel = os.path.relpath(fpath)
                    results.append(f"{rel}:{j + 1}:{prefix}\t{lines[j].rstrip()}")
                    line_count += 1
                results.append("--")
            else:
                rel = os.path.relpath(fpath)
                results.append(f"{rel}:{i + 1}:\t{lines[i].rstrip()}")
                line_count += 1

        if head_limit is not None and line_count >= head_limit:
            break

    return {
        "success": True,
        "mode": "content",
        "num_lines": line_count,
        "content": "\n".join(results),
        "truncated": head_limit is not None and line_count >= head_limit
    }
