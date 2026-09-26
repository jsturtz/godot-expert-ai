import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import chromadb
from mcp.server import MCPServer

from ingest import run_ingestion_pipeline

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
CHROMA_PATH = CACHE_DIR / "chroma_db"

async def periodic_ingestion_task(interval_seconds: int = 3600):
    """Seed the index immediately on startup and then refresh periodically."""
    try:
        print("[BACKGROUND TASK] Initial Godot docs sync and index seed...")
        await asyncio.to_thread(run_ingestion_pipeline)
        print("[BACKGROUND TASK] Initial ingestion complete.")
    except Exception as e:
        print(f"[BACKGROUND TASK ERROR] Failed during initial ingest: {e}")

    while True:
        try:
            print("[BACKGROUND TASK] Checking for Godot doc updates...")
            await asyncio.to_thread(run_ingestion_pipeline)
            print("[BACKGROUND TASK] Ingestion check complete.")
        except Exception as e:
            print(f"[BACKGROUND TASK ERROR] Failed to ingest: {e}")

        await asyncio.sleep(interval_seconds)

# Use lifespan context manager to start background loop on server startup
@asynccontextmanager
async def server_lifespan(server: MCPServer):
    """Lifespan context manager to start background ingestion task."""
    # Start background updater task
    task = asyncio.create_task(periodic_ingestion_task(3600))
    yield
    # Cleanup on shutdown
    task.cancel()

mcp = MCPServer("Godot ChromaDB Search Server", lifespan=server_lifespan)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
godot_docs_collection = chroma_client.get_collection("godot_docs")

# --- TOOLS ---
@mcp.tool()
async def search_godot_docs(
    query: str,
    language: Optional[Literal["gdscript", "csharp", "cpp"]] = None,
    n_results: int = 5,
) -> list[str]:
    """
    Search Godot engine documentation.

    Args:
        query: Natural-language question about Godot.
        language: Restrict code examples to this language. Omit to search
            all content regardless of language (recommended default —
            only set this when the user explicitly asks for a specific
            language's syntax).
        n_results: Number of chunks to return.
    """
    where = {"language": {"$in": [language, "any"]}} if language else None
    results = godot_docs_collection.query(query_texts=[query], n_results=n_results, where=where)
    return results["documents"][0]

# --- PROMPTS ---
@mcp.prompt()
def godot_code_assistant_prompt(topic: str) -> str:
    """Returns a pre-configured prompt template for generating GDScript code."""
    return f"You are an expert Godot 4 developer. Write a clean GDScript snippet demonstrating: {topic}"

# --- RESOURCES ---
@mcp.resource("resource://godot/version")
def get_godot_version() -> str:
    """Returns the current target Godot version."""
    return "Godot Engine v4.7.stable"

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    mcp.run(transport="sse", host=host, port=port)
