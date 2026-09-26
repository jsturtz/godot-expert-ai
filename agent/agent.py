
import os
from typing import TypedDict

from deepagents import create_deep_agent
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

OLLAMA_MODEL = "qwen3:14b"  # Ollama model for routing and task execution


class MessagesState(TypedDict):
    """State structure for KK's LangGraph workflow."""
    user_message: str
    requires_godot_expertise: bool

def get_mcp_client() -> MultiServerMCPClient:
    """
    Initialize and return a MultiServerMCPClient for interacting with MCP servers.
    """
    return MultiServerMCPClient({
        "godot_docs": {
            "transport": "sse",
            "url": os.environ.get("GODOT_MCP_SERVER", "http://localhost:8000/sse"),
        }
    })

async def agent_node(state: MessagesState) -> dict:
    """FIXME"""
    # requires_godot_expertise = state.get("requires_godot_expertise", False)

    model = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    mcp_client = get_mcp_client()
    mcp_tools = await mcp_client.get_tools()

    # Could pass memory to create_deep_agent, but then it would only apply to this node...interesting...
    task_agent = create_deep_agent(
        model=model,
        tools=mcp_tools,
        system_prompt=(
            "You are a Godot expert."
            "For factual questions about Godot APIs, setup, or behavior, "
            "always use the Godot docs search tool before answering. "
            "Prefer documented answers over model memory. "
            "If the tool fails, say so and explain the limitation."
        ),
    )

    try:
        result = await task_agent.ainvoke({"messages": [("user", state["user_message"])]})
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            task_result = getattr(last_message, "content", str(last_message))
        else:
            task_result = "Task completed without output."
    except Exception as e:
        task_result = f"Encountered an issue executing task: {e}"

    return {"task_result": task_result}


# ==========================================
# 6. Build the LangGraph Workflow
# ==========================================

def create_agent_graph():
    """Build and compile the KK StateGraph."""
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    return workflow.compile(
        # checkpointer=InMemorySaver(),
        # store=InMemoryStore(),
    )

graph = create_agent_graph()