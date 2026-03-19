"""
Database query tool for agents.

Provides read-only SQL query execution against SQLite databases.
Extend for PostgreSQL, BigQuery, etc. as needed.
"""

import sqlite3
from pathlib import Path

from agents.tools.base_tool import agent_tool


@agent_tool(
    name="database",
    description=(
        "Execute a read-only SQL query against a SQLite database. "
        "Returns results as a list of dictionaries."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute (SELECT only)",
            },
            "db_path": {
                "type": "string",
                "description": "Path to the SQLite database file",
            },
        },
        "required": ["query", "db_path"],
    },
)
def database_query(query: str, db_path: str) -> list[dict]:
    """Execute a read-only SQL query."""
    query_stripped = query.strip().upper()
    if not query_stripped.startswith("SELECT") and not query_stripped.startswith("WITH"):
        raise ValueError("Only SELECT and WITH (CTE) queries are allowed")

    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    finally:
        conn.close()
