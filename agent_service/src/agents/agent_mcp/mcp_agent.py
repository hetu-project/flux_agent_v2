"""MCP Agent - Agent that uses MCP service for tool calling instead of function calling."""

from typing import List, Dict, Any, Optional
import json
from openai import OpenAI
from src.config import get_settings
from src.services.mcp_service import MCPService
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class MCPAgent:
    """Agent that uses MCP service for tool calling."""
    
    def __init__(self):
        self.mcp_service = MCPService()
        
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
        
        prompt = f"""你是一个智能助手，可以使用以下工具来帮助用户：

可用工具：
{tools_description}

用户问题：{user_question}

请分析用户问题，并选择最合适的工具来回答。请以 JSON 格式返回：
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
                    {"role": "system", "content": "你是一个智能助手，可以分析用户问题并选择合适的工具。只返回 JSON 格式的工具选择结果。"},
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
        使用 MCP 服务处理用户查询。
        
        流程：
        1. 获取所有可用的 MCP 工具
        2. 让 LLM 分析用户问题并选择合适的工具
        3. 如果选择了工具，通过 MCP 服务调用工具
        4. 将工具结果和用户问题一起发送给 LLM 生成最终回答
        
        Args:
            user_question: 用户问题
            top_k: 保留参数（用于兼容性）
            api_key: Optional API key (if provided, uses OpenRouter)
            base_url: Optional base URL (if provided, uses this URL)
            
        Returns:
            Agent response with answer
        """
        logger.info(f"Processing MCP agent query: {user_question[:100]}...")
        
        # 获取 LLM 客户端
        llm_client = self._get_llm_client(api_key=api_key, base_url=base_url)
        model_name = settings.openrouter_model if api_key else self.chat_model
        
        try:
            # 步骤 1: 获取所有可用工具
            logger.debug("Getting available MCP tools...")
            tools = await self._get_tools()
            logger.info(f"MCP tools retrieved: {len(tools)} tools available")
            
            if not tools:
                logger.warning("No MCP tools available")
                # 如果没有工具，直接使用 LLM 回答
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "你是一个智能助手。"},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.7,
                    max_tokens=300  # Limit to ~200 English words
                )
                return {
                    "answer": response.choices[0].message.content,
                    "sources": [],
                    "num_sources": 0
                }
            
            # 步骤 2: 让 LLM 选择工具
            logger.debug("Asking LLM to choose tool...")
            tool_choice = await self._llm_choose_tool(
                user_question,
                tools,
                llm_client,
                model_name
            )
            if tool_choice:
                logger.info(f"LLM tool choice: {json.dumps(tool_choice, indent=2, ensure_ascii=False)}")
            else:
                logger.info("LLM determined no tool needed")
            
            # 步骤 3: 如果选择了工具，调用工具
            tool_result = None
            tool_result_data = None
            if tool_choice and tool_choice.get("tool_name"):
                tool_name = tool_choice["tool_name"]
                tool_arguments = tool_choice.get("arguments", {})
                
                logger.info(f"Calling MCP tool: {tool_name} with arguments: {tool_arguments}")
                tool_result = await self.mcp_service.call_tool(tool_name, tool_arguments)
                
                # 提取工具结果
                if hasattr(tool_result, 'data') and tool_result.data:
                    tool_result_data = tool_result.data
                elif hasattr(tool_result, 'structured_content') and tool_result.structured_content:
                    tool_result_data = tool_result.structured_content
                else:
                    tool_result_dict = tool_result.model_dump() if hasattr(tool_result, 'model_dump') else tool_result.dict() if hasattr(tool_result, 'dict') else {}
                    tool_result_data = tool_result_dict
                
                # 记录 MCP 响应
                logger.info(f"MCP tool {tool_name} response: {json.dumps(tool_result_data, indent=2, ensure_ascii=False, default=str)}")
                logger.info(f"Tool {tool_name} returned result successfully")
            else:
                logger.debug("No tool selected, will use LLM directly")
            
            # 步骤 4: 将工具结果和用户问题一起发送给 LLM
            if tool_result_data:
                prompt = f"""用户问题：{user_question}

工具调用结果：
{json.dumps(tool_result_data, indent=2, ensure_ascii=False, default=str)}

请根据工具返回的结果，生成一个清晰、有用的回答。"""
            else:
                prompt = user_question
            
            logger.debug(f"Generating LLM response (model: {model_name})")
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个智能助手，根据工具返回的结果回答用户问题。如果工具返回了结果，请基于这些结果生成回答；如果没有工具结果，请直接回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300  # Limit to ~200 English words
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response (length: {len(answer)})")
            
            return {
                "answer": answer,
                "sources": [],
                "num_sources": 0,
                "tool_used": tool_choice.get("tool_name") if tool_choice else None,
                "tool_result": tool_result_data if tool_result_data else None
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise
    
    async def close(self):
        """关闭 MCP 服务连接"""
        await self.mcp_service.close()

