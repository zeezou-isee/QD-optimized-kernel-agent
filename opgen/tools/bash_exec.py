"""
BashExec tool - Execute bash commands in a sandboxed environment.

Inspired by Claude Code's BashTool. Executes shell commands with timeout
and returns stdout, stderr, and exit code.
"""

import subprocess
import os
from typing import Any, Optional


TOOL_NAME = "bash_exec"

SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Execute a bash command and return its output. Use for running build commands, tests, git operations, and other shell tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds (max 600000). Defaults to 120000 (2 minutes)."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to current working directory."
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables to set.",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["command"]
        }
    }
}

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000

# Commands that require extra caution
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "mkfs.",
    "dd if=",
    ":(){ :|:& };:",  # fork bomb
]


def bash_exec(
    command: str,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None
) -> dict[str, Any]:
    """
    Execute a bash command.

    Args:
        command: The shell command to run.
        timeout: Timeout in milliseconds (max 600000). Defaults to 120000.
        cwd: Working directory for the command.
        env: Additional environment variables.

    Returns:
        A dict with stdout, stderr, exit_code, and timing info.
    """
    # Safety check
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return {
                "success": False,
                "error": f"Command rejected: dangerous pattern detected ('{pattern}')."
            }

    timeout_ms = min(timeout or DEFAULT_TIMEOUT_MS, MAX_TIMEOUT_MS)
    timeout_sec = timeout_ms / 1000.0

    # Build environment
    exec_env = os.environ.copy()
    if env:
        exec_env.update(env)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=cwd or os.getcwd(),
            env=exec_env,
            executable="/bin/bash"
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout_ms}ms",
            "exit_code": -1,
            "stdout": "",
            "stderr": ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Command execution failed: {e}",
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e)
        }

    return {
        "success": result.returncode == 0,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
        "cwd": cwd or os.getcwd()
    }
