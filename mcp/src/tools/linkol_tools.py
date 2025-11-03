"""Linkol-related MCP tools using FastMCP decorators."""

from typing import Dict, Any
from utils.logger import get_logger
from core.services import (
    get_linkol_service,
    get_project_content_repo,
    get_qdrant_service,
    get_embedding_service
)

logger = get_logger(__name__)


# Note: Tools are registered in server.py using @mcp.tool() decorator
# This file contains the tool implementations


async def get_kol_price_impl(screen_name: str) -> Dict[str, Any]:
    """Implementation for get_kol_price tool."""
    logger.info(f"Getting KOL price for @{screen_name}")
    linkol_service = get_linkol_service()
    
    try:
        result = await linkol_service.get_kol_price(screen_name=screen_name)
        
        if result.get("code") == 200:
            price = result.get("data", {}).get("price")
            return {
                "success": True,
                "screen_name": screen_name,
                "price": price,
                "message": f"@{screen_name} 的当前估值是 ${price:.2f}"
            }
        else:
            return {
                "success": False,
                "error": result.get("msg", "获取估值失败"),
                "code": result.get("code")
            }
    except Exception as e:
        logger.error(f"Error getting KOL price: {e}")
        return {"success": False, "error": str(e)}


async def get_hot_kols_impl(limit: int = 20) -> Dict[str, Any]:
    """Implementation for get_hot_kols tool."""
    logger.info(f"Getting hot KOLs (limit: {limit})")
    linkol_service = get_linkol_service()
    
    try:
        result = await linkol_service.get_hot_kols()
        
        if result.get("code") == 200:
            kols = result.get("data", {}).get("list", [])[:limit]
            return {
                "success": True,
                "kols": kols,
                "count": len(kols),
                "message": f"获取到 {len(kols)} 个热门KOL"
            }
        else:
            return {
                "success": False,
                "error": result.get("msg", "获取KOL列表失败"),
                "code": result.get("code")
            }
    except Exception as e:
        logger.error(f"Error getting hot KOLs: {e}")
        return {"success": False, "error": str(e)}


async def search_linkol_content_impl(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Implementation for search_linkol_content tool."""
    logger.info(f"Searching Linkol content: {query}")
    project_content_repo = get_project_content_repo()
    
    if not project_content_repo:
        return {"error": "Project content repository not available"}
    
    try:
        results = project_content_repo.search(
            query=query,
            project_name="Linkol",
            top_k=top_k,
            min_score=0.6
        )
        
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "message": f"找到 {len(results)} 条相关内容"
        }
    except Exception as e:
        logger.error(f"Error searching content: {e}")
        return {"success": False, "error": str(e)}


async def check_other_projects_impl(user_question: str) -> Dict[str, Any]:
    """Implementation for check_other_projects tool using vector similarity search."""
    logger.info(f"Checking other projects in question: {user_question[:50]}...")
    project_content_repo = get_project_content_repo()
    qdrant_service = get_qdrant_service()
    embedding_service = get_embedding_service()
    
    if not project_content_repo:
        return {
            "has_other_project": False,
            "project_name": None
        }
    
    try:
        # Get all unique project names first (excluding Linkol)
        collection_name = "project_content"
        
        project_names = set()
        offset = None
        
        while True:
            result = qdrant_service.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset
            )
            
            points, next_offset = result
            
            if not points:
                break
            
            for point in points:
                payload = point.payload
                project_name = payload.get("project_name")
                if project_name and project_name != "Linkol":
                    project_names.add(project_name)
            
            if next_offset is None:
                break
            offset = next_offset
        
        if not project_names:
            logger.debug("No other projects found in database")
            return {
                "has_other_project": False,
                "project_name": None
            }
        
        # Use vector similarity search instead of string matching
        from qdrant_client.http.models import Filter, FieldCondition, MatchAny
        
        # Generate embedding for user question
        query_vector = embedding_service.embed_text(user_question)
        
        # Create filter to exclude Linkol and only search other projects
        filter_query = Filter(
            must=[
                FieldCondition(
                    key="project_name",
                    match=MatchAny(any=list(project_names))  # Only search non-Linkol projects
                )
            ]
        )
        
        # Search for most similar content
        results = qdrant_service.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,  # Top 5 results
            filter_query=filter_query
        )
        
        if not results:
            logger.debug("No similar project content found for user question")
            return {
                "has_other_project": False,
                "project_name": None
            }
        
        # Filter results by minimum similarity threshold (0.7 for cosine similarity)
        min_score = 0.7
        filtered_results = [r for r in results if r.score >= min_score]
        
        if not filtered_results:
            logger.debug(f"Top result score: {results[0].score:.3f} (below threshold {min_score})")
            return {
                "has_other_project": False,
                "project_name": None
            }
        
        # Get the top result and check its project name
        top_result = filtered_results[0]
        top_project = top_result.payload.get("project_name")
        
        # Only return if similarity score is high enough and project is not Linkol
        if top_project and top_project != "Linkol":
            logger.info(f"Detected semantically similar project: {top_project} (score: {top_result.score:.3f})")
            return {
                "has_other_project": True,
                "project_name": top_project,
                "score": top_result.score,
                "message": f"Questions about {top_project} should be directed to our parallel universe agent. Please visit the parallel universe agent for queries regarding {top_project}."
            }
        
        return {
            "has_other_project": False,
            "project_name": None
        }
        
    except Exception as e:
        logger.error(f"Error checking other projects: {e}")
        return {
            "has_other_project": False,
            "project_name": None,
            "error": str(e)
        }
