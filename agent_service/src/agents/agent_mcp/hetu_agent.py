"""Hetu Agent using MCP service for tool calling instead of function calling."""

from typing import List, Dict, Any, Optional
import json
from openai import OpenAI
from src.config import get_settings
from src.services.mcp_service import MCPService
from src.repositories.project_repository import ProjectRepository
from src.repositories.project_content_repository import ProjectContentRepository
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class HetuMCPAgent:
    """Hetu Agent that uses MCP service for tool calling instead of function calling."""
    
    PROJECT_NAME = "Hetu Protocol"  # Fixed project name
    
    def __init__(
        self,
        project_repo: Optional[ProjectRepository] = None,
        project_content_repo: Optional[ProjectContentRepository] = None
    ):
        self.mcp_service = MCPService()
        self.project_repo = project_repo
        self.project_content_repo = project_content_repo
        
        # Initialize default LLM client with AIHubMix
        if not settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is required. Please set it in .env file or environment variables.")
        
        self.llm = OpenAI(
            api_key=settings.aihubmix_api_key,
            base_url=settings.aihubmix_base_url
        )
        self.chat_model = settings.chat_model
        self._cached_tools: Optional[List[Dict[str, Any]]] = None
    
    def _get_llm_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
        """
        Get LLM client with optional API key and base URL.
        If provided, uses OpenRouter; otherwise uses default AIHubMix.
        
        Args:
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            OpenAI client instance
        """
        if api_key:
            # Use OpenRouter if API key is provided
            client_base_url = base_url or settings.openrouter_base_url
            return OpenAI(
                api_key=api_key,
                base_url=client_base_url
            )
        else:
            # Use default AIHubMix
            return self.llm
    
    def _get_hetu_project(self) -> Optional[Dict[str, Any]]:
        """
        Get Hetu Protocol project from database.
        
        Returns:
            Project dict with name and description if found, None otherwise
        """
        if not self.project_repo:
            logger.debug("Project repository not available")
            return None
        
        try:
            logger.debug(f"Getting project: {self.PROJECT_NAME}")
            project_obj = self.project_repo.get_by_name(self.PROJECT_NAME)
            
            if project_obj:
                project = {
                    "name": project_obj.name,
                    "description": project_obj.description
                }
                logger.info(f"Found project: {self.PROJECT_NAME}")
                return project
            else:
                logger.warning(f"Project '{self.PROJECT_NAME}' not found in database")
                return None
        except Exception as e:
            logger.error(f"Error getting Hetu project: {e}")
            return None
    
    async def _get_tools(self) -> List[Dict[str, Any]]:
        """
        获取所有可用的 MCP 工具（带缓存）。
        
        Returns:
            工具列表
        """
        if self._cached_tools is None:
            tools = await self.mcp_service.list_tools()
            # 将 Tool 对象转换为字典
            self._cached_tools = []
            for tool in tools:
                tool_dict = {
                    "name": getattr(tool, 'name', 'Unknown'),
                    "description": getattr(tool, 'description', 'No description'),
                    "inputSchema": {}
                }
                # 提取参数信息
                input_schema = getattr(tool, 'inputSchema', None)
                if input_schema:
                    tool_dict["inputSchema"] = input_schema.model_dump() if hasattr(input_schema, 'model_dump') else input_schema.dict() if hasattr(input_schema, 'dict') else {}
                self._cached_tools.append(tool_dict)
            logger.info(f"Cached {len(self._cached_tools)} MCP tools")
        
        return self._cached_tools
    
    def _format_tools_for_llm(self, tools: List[Dict[str, Any]]) -> str:
        """
        格式化工具列表，让 LLM 更容易理解。
        
        Args:
            tools: 工具列表
            
        Returns:
            格式化后的工具描述字符串
        """
        formatted = []
        for tool in tools:
            tool_info = f"""
工具名称：{tool.get('name', 'Unknown')}
描述：{tool.get('description', 'No description')}
参数：
"""
            input_schema = tool.get('inputSchema', {})
            properties = input_schema.get('properties', {})
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', 'No description')
                required = param_info.get('required', False)
                default = param_info.get('default', None)
                
                param_str = f"  - {param_name} ({param_type})"
                if required:
                    param_str += " [必需]"
                if default is not None:
                    param_str += f" [默认: {default}]"
                param_str += f": {param_desc}"
                tool_info += param_str + "\n"
            
            formatted.append(tool_info)
        
        return "\n".join(formatted)
    
    async def _llm_choose_tool(
        self,
        user_question: str,
        tools: List[Dict[str, Any]],
        llm_client: OpenAI,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        让 LLM 分析用户问题并选择合适的工具。
        
        Args:
            user_question: 用户问题
            tools: 可用工具列表
            llm_client: LLM 客户端
            model_name: 模型名称
            
        Returns:
            工具选择结果，包含工具名称和参数，如果不需要工具则返回 None
        """
        tools_description = self._format_tools_for_llm(tools)
        
        prompt = f"""你是一个智能助手，专门回答关于 Hetu Protocol 的问题。你可以使用以下工具来帮助用户：

可用工具：
{tools_description}

用户问题：{user_question}

请分析用户问题，并选择最合适的工具来回答关于 Hetu Protocol 的问题。请以 JSON 格式返回：
{{
    "tool_name": "工具名称",
    "arguments": {{"参数名": "参数值"}}
}}

如果不需要使用工具，返回 {{"tool_name": null}}

只返回 JSON，不要其他内容。"""
        
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个智能助手，专门回答关于 Hetu Protocol 的问题。你可以分析用户问题并选择合适的工具。只返回 JSON 格式的工具选择结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 尝试解析 JSON
            # 移除可能的 markdown 代码块标记
            if answer.startswith("```"):
                # 提取 JSON 部分
                lines = answer.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_json = not in_json
                        continue
                    if in_json:
                        json_lines.append(line)
                answer = "\n".join(json_lines)
            
            tool_choice = json.loads(answer)
            
            if tool_choice.get("tool_name") is None:
                logger.debug("LLM determined no tool needed")
                return None
            
            logger.info(f"LLM chose tool: {tool_choice['tool_name']} with arguments: {tool_choice.get('arguments', {})}")
            return tool_choice
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}, response: {answer}")
            return None
        except Exception as e:
            logger.error(f"Error in LLM tool selection: {e}", exc_info=True)
            return None
    
    async def query(
        self,
        user_question: str,
        top_k: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the agent with a question about Hetu Protocol using MCP tools.
        
        Logic:
        1. Always use Hetu Protocol project information
        2. Get available MCP tools
        3. Let LLM choose appropriate tools to call
        4. Call MCP tools to get information
        5. Search for relevant content related to Hetu Protocol (via MCP or direct)
        6. Include project info and content in the prompt
        7. Use LLM to generate answer with Hetu Protocol introducer role
        
        Args:
            user_question: User's question
            top_k: Number of relevant documents to retrieve
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Agent response with answer and sources
        """
        logger.info(f"Processing Hetu Protocol query with MCP: {user_question[:100]}...")
        
        # 1. Always get Hetu Protocol project
        found_project = self._get_hetu_project()
        
        # 2. Get LLM client
        llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
        model_name = settings.openrouter_model if api_key else self.chat_model
        
        # 3. Search for relevant content from project_content collection
        sources = []
        relevant_content = []
        
        if found_project and self.project_content_repo:
            # Search for relevant content related to Hetu Protocol
            logger.debug(f"Searching for relevant content for project '{self.PROJECT_NAME}'")
            relevant_content = self.project_content_repo.search(
                query=user_question,
                project_name=self.PROJECT_NAME,
                top_k=top_k,
                min_score=0.6  # Minimum similarity threshold
            )
            
            # Format sources from relevant content
            for item in relevant_content[:top_k]:  # Limit to top_k
                source_info = {
                    "type": item.get("content_type", "unknown"),
                    "content": item.get("content", "")[:200] + "..." if len(item.get("content", "")) > 200 else item.get("content", ""),
                    "score": item.get("score", 0)
                }
                if item.get("title"):
                    source_info["title"] = item.get("title")
                if item.get("author"):
                    source_info["author"] = item.get("author")
                if item.get("source_url"):
                    source_info["url"] = item.get("source_url")
                sources.append(source_info)
            
            logger.info(f"Found {len(sources)} relevant content items for project '{self.PROJECT_NAME}'")
        
        # 4. Get available MCP tools and let LLM choose
        tool_results = []
        try:
            logger.debug("Getting available MCP tools...")
            tools = await self._get_tools()
            logger.info(f"MCP tools retrieved: {len(tools)} tools available")
            
            if tools:
                # Let LLM choose tools to call
                logger.debug("Asking LLM to choose tool...")
                tool_choice = await self._llm_choose_tool(
                    user_question,
                    tools,
                    llm_client,
                    model_name
                )
                
                # Call MCP tool if selected
                if tool_choice and tool_choice.get("tool_name"):
                    tool_name = tool_choice["tool_name"]
                    tool_arguments = tool_choice.get("arguments", {})
                    
                    logger.info(f"Calling MCP tool: {tool_name} with arguments: {tool_arguments}")
                    tool_result = await self.mcp_service.call_tool(tool_name, tool_arguments)
                    
                    # Extract tool result
                    if hasattr(tool_result, 'data') and tool_result.data:
                        tool_result_data = tool_result.data
                    elif hasattr(tool_result, 'structured_content') and tool_result.structured_content:
                        tool_result_data = tool_result.structured_content
                    else:
                        tool_result_dict = tool_result.model_dump() if hasattr(tool_result, 'model_dump') else tool_result.dict() if hasattr(tool_result, 'dict') else {}
                        tool_result_data = tool_result_dict
                    
                    tool_results.append({
                        "tool_name": tool_name,
                        "result": tool_result_data
                    })
                    
                    logger.info(f"MCP tool {tool_name} returned result successfully")
        except Exception as e:
            logger.warning(f"Error using MCP tools: {e}", exc_info=True)
            # Continue without MCP tools if there's an error
        
        # 5. Build prompt with Hetu Protocol information
        if found_project:
            # Include project information in the prompt
            project_info = f"Project name: {found_project['name']}"
            if found_project.get('description'):
                project_info += f"\nProject description: {found_project['description']}"
            
            # Add relevant content context if available
            content_context = ""
            if relevant_content:
                content_context = "\n\nRelevant content from tweets, papers, and other sources:\n"
                for i, item in enumerate(relevant_content[:top_k], 1):
                    content_type = item.get("content_type", "content")
                    content_text = item.get("content", "")[:300]  # Limit length
                    if item.get("title"):
                        content_context += f"\n[{i}] {item['title']} ({content_type}): {content_text}\n"
                    else:
                        content_context += f"\n[{i}] ({content_type}): {content_text}\n"
            
            # Add MCP tool results if available
            mcp_context = ""
            if tool_results:
                mcp_context = "\n\nInformation from MCP tools:\n"
                for i, tool_result in enumerate(tool_results, 1):
                    tool_name = tool_result.get("tool_name", "Unknown")
                    result_data = tool_result.get("result", {})
                    mcp_context += f"\n[{i}] Tool: {tool_name}\nResult: {json.dumps(result_data, indent=2, ensure_ascii=False, default=str)}\n"
            
            prompt = f"""Answer the user's question about Hetu Protocol based on the following project information, relevant content, and MCP tool results.

{project_info}{content_context}{mcp_context}

User question: {user_question}

Please provide a helpful answer about Hetu Protocol based on the project info, relevant content, and MCP tool results above. If the information is insufficient, please say so."""
        else:
            # Add MCP tool results even if project not found
            mcp_context = ""
            if tool_results:
                mcp_context = "\n\nInformation from MCP tools:\n"
                for i, tool_result in enumerate(tool_results, 1):
                    tool_name = tool_result.get("tool_name", "Unknown")
                    result_data = tool_result.get("result", {})
                    mcp_context += f"\n[{i}] Tool: {tool_name}\nResult: {json.dumps(result_data, indent=2, ensure_ascii=False, default=str)}\n"
            
            prompt = f"""Answer the user's question about Hetu Protocol based on the following MCP tool results.{mcp_context}

User question: {user_question}

Please provide a helpful answer about Hetu Protocol based on the MCP tool results above."""
        
        # 6. Generate answer using LLM with Hetu Protocol introducer role
        logger.debug(f"Generating LLM response (model: {model_name})")
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable Hetu Protocol introducer who helps users learn about Hetu Protocol. Your role is to introduce and explain Hetu Protocol in a professional yet friendly manner. You focus on explaining what Hetu Protocol is, how it works, its key features, technical details, and applications. Your tone is clear, informative, and approachable. You are enthusiastic about Hetu Protocol but maintain professionalism. Always prioritize providing accurate information about Hetu Protocol based on the provided context and MCP tool results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300  # Limit to ~200 English words
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response (length: {len(answer)})")
            
            return {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources),
                "tools_used": [tr.get("tool_name") for tr in tool_results] if tool_results else None
            }
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise
    
    async def close(self):
        """关闭 MCP 服务连接"""
        await self.mcp_service.close()

