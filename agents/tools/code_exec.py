"""
Code execution tool for agents.

Executes Python code in a restricted subprocess. For production,
replace with a proper sandbox (Docker container, AWS Lambda, etc.).
"""

import subprocess
import tempfile
from pathlib import Path

from agents.tools.base_tool import agent_tool


@agent_tool(
    name="code_exec",
    description=(
        "Execute Python code and return stdout/stderr. "
        "Use for data processing, calculations, and analysis. "
        "Code runs in a temporary file with a 30-second timeout."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
        },
        "required": ["code"],
    },
)
def code_exec(code: str) -> dict:
    """Execute Python code in a subprocess."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=tempfile.gettempdir(),
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Execution timed out after 30 seconds",
            "returncode": -1,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
