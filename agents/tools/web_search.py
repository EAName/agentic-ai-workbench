"""
Web search tool for agents.

Performs lightweight web search via DuckDuckGo HTML results and returns
top result snippets as plain text.
"""

from __future__ import annotations

from urllib.parse import quote

import aiohttp

from agents.tools.base_tool import agent_tool


@agent_tool(
    name="web_search",
    description="Search the web and return top result snippets for a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
async def web_search(query: str, max_results: int = 5) -> dict:
    """Search the public web and return a compact list of results."""
    limit = max(1, min(int(max_results), 10))
    url = f"https://duckduckgo.com/html/?q={quote(query)}"
    headers = {"User-Agent": "agentic-ai-workbench/1.0 (+https://duckduckgo.com)"}

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError(f"Web search failed with status {response.status}")
            html = await response.text()

    results: list[dict] = []
    chunks = html.split('<a rel="nofollow" class="result__a"')
    for chunk in chunks[1:]:
        href_start = chunk.find('href="')
        title_start = chunk.find(">")
        title_end = chunk.find("</a>")
        if href_start == -1 or title_start == -1 or title_end == -1:
            continue

        href_start += len('href="')
        href_end = chunk.find('"', href_start)
        link = chunk[href_start:href_end]
        title = chunk[title_start + 1 : title_end].strip()

        snippet = ""
        snippet_key = 'class="result__snippet"'
        snippet_pos = chunk.find(snippet_key)
        if snippet_pos != -1:
            snippet_open = chunk.find(">", snippet_pos)
            snippet_close = chunk.find("</a>", snippet_open)
            if snippet_close == -1:
                snippet_close = chunk.find("</div>", snippet_open)
            if snippet_open != -1 and snippet_close != -1:
                snippet = chunk[snippet_open + 1 : snippet_close].strip()

        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet})
        if len(results) >= limit:
            break

    return {"query": query, "results": results}
