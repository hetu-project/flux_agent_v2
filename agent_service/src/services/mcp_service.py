"""MCP (Model Context Protocol) service for calling MCP server tools using FastMCP client."""

from typing import Optional, Dict, Any, List
from fastmcp import Client
from src.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class MCPService:
    """Service for interacting with MCP server using FastMCP client."""
    
    def __init__(self):
        self.base_url = settings.mcp_url.rstrip('/')
        self.endpoint = settings.mcp_endpoint
        self.full_url = f"{self.base_url}{self.endpoint}"
        self.client: Optional[Client] = None
        self.initialized = False
    
    async def _ensure_client(self):
        """确保客户端已创建并初始化"""
        if self.client is None:
            logger.info(f"Creating MCP client connection to {self.full_url}")
            # FastMCP Client 需要服务器 URL
            self.client = Client(self.full_url)
            # 初始化客户端连接
            await self.client.__aenter__()
            self.initialized = True
            logger.info("MCP client connected successfully")
    
    async def initialize(self) -> Dict[str, Any]:
        """
        初始化 MCP 会话。
        
        根据 MCP 协议，需要先调用 initialize 方法来建立会话。
        
        Returns:
            初始化响应，包含服务器信息和能力
        """
        await self._ensure_client()
        
        try:
            logger.info("Initializing MCP session...")
            # FastMCP Client 会自动处理 initialize
            # 如果需要手动初始化，可以调用相应的方法
            result = {
                "initialized": True,
                "url": self.full_url
            }
            logger.info("MCP session initialized successfully")
            return result
        except Exception as e:
            logger.error(f"Error initializing MCP session: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有可用的工具。
        
        Returns:
            工具列表，每个工具包含名称、描述、参数等信息
        """
        await self._ensure_client()
        
        try:
            logger.debug("Listing MCP tools...")
            # 使用 FastMCP Client 的 list_tools 方法
            tools = await self.client.list_tools()
            logger.info(f"Found {len(tools)} MCP tools")
            return tools
        except Exception as e:
            logger.error(f"Error listing tools: {e}", exc_info=True)
            return []
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        调用 MCP 工具。
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
        """
        await self._ensure_client()
        
        try:
            logger.info(f"Calling MCP tool: {tool_name} with arguments: {arguments}")
            # 使用 FastMCP Client 的 call_tool 方法
            result = await self.client.call_tool(tool_name, arguments)
            logger.info(f"MCP tool {tool_name} executed successfully")
            return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def close(self):
        """关闭客户端连接"""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
                logger.info("MCP client closed")
            except Exception as e:
                logger.error(f"Error closing MCP client: {e}")
            finally:
                self.client = None
                self.initialized = False
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
