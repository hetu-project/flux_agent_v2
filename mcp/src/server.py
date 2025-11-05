"""MCP Server implementation using FastMCP."""

from fastmcp import FastMCP
import sys
from pathlib import Path
from typing import Dict, Any

# Setup import paths
mcp_src = Path(__file__).parent
project_root = mcp_src.parent.parent
sys.path.insert(0, str(mcp_src))
sys.path.insert(0, str(project_root))

from utils.logger import setup_logging, get_logger
from core.services import init_services
from tools.linkol_tools import (
    get_kol_price_impl,
    get_hot_kols_impl,
    search_linkol_content_impl,
    check_other_projects_impl
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastMCP
mcp = FastMCP("Hetu Agent MCP Server")

# Initialize services at startup
init_services()
logger.info("MCP Server initialized")

# ============================================================================
# MCP Tools using @mcp.tool() decorator
# ============================================================================

@mcp.tool()
async def get_kol_price(screen_name: str) -> Dict[str, Any]:
    """
    Get the KOL valuation price for a specific Twitter user.
    
    Call this tool when the user asks about the valuation of a specific user (e.g., @username).
    
    Args:
        screen_name: Twitter username without @ symbol, e.g., vis_eth or dada81505550664
    
    Returns:
        Dictionary containing price information
    """
    return await get_kol_price_impl(screen_name)


@mcp.tool()
async def get_hot_kols(limit: int = 20) -> Dict[str, Any]:
    """
    Get the list of hot KOLs.
    
    Call this tool when the user asks about hot KOLs, rankings, top KOLs, or KOL leaderboard.
    
    Args:
        limit: Number of KOLs to return, default 20
    
    Returns:
        Dictionary containing KOL list
    """
    return await get_hot_kols_impl(limit)


@mcp.tool()
async def search_linkol_content(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search for relevant content in the Linkol project knowledge base.
    
    Call this tool when the user asks about what Linkol is, how to use it, introduction, etc.
    
    Args:
        query: Search query text
        top_k: Number of results to return, default 5
    
    Returns:
        Dictionary containing search results
    """
    return await search_linkol_content_impl(query, top_k)


@mcp.tool()
async def check_other_projects(user_question: str) -> Dict[str, Any]:
    """
    Check if user's question mentions other projects (excluding Linkol) using vector similarity search.
    
    Uses semantic similarity to detect if the question is about another project, and if so,
    should direct the user to visit the parallel universe agent.
    
    Args:
        user_question: The complete user question text
    
    Returns:
        Dictionary containing check results with project name and message if found
    """
    return await check_other_projects_impl(user_question)


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting MCP Server in HTTP mode...")
    mcp.run(transport="http", host="0.0.0.0", port=6001)