"""
File operations tool for agents.

Allows agents to read, write, and list files in the workspace.
"""

from pathlib import Path

from agents.tools.base_tool import agent_tool


@agent_tool(
    name="read_file",
    description="Read the contents of a file. Returns the file text.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read",
            }
        },
        "required": ["file_path"],
    },
)
def read_file(file_path: str) -> str:
    """Read a file and return its contents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8")


@agent_tool(
    name="write_file",
    description="Write content to a file. Creates the file if it does not exist.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    },
)
def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {file_path}"


@agent_tool(
    name="list_directory",
    description="List files and directories at a given path.",
    parameters={
        "type": "object",
        "properties": {
            "directory_path": {
                "type": "string",
                "description": "Path to the directory to list",
            }
        },
        "required": ["directory_path"],
    },
)
def list_directory(directory_path: str) -> list[str]:
    """List contents of a directory."""
    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    return sorted([str(p.relative_to(path)) for p in path.iterdir()])
