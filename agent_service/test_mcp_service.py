#!/usr/bin/env python3
"""测试 MCP 服务连接"""

import asyncio
import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.services.mcp_service import MCPService
from src.utils.logger import setup_logging, get_logger
from src.config import get_settings

# 设置日志
setup_logging()
logger = get_logger(__name__)
settings = get_settings()


async def test_mcp_service():
    """测试 MCP 服务"""
    
    print("=" * 70)
    print("测试 MCP 服务连接")
    print("=" * 70)
    print(f"MCP 服务器地址: {settings.mcp_url}{settings.mcp_endpoint}")
    print()
    
    # 创建 MCP 服务实例
    mcp_service = MCPService()
    
    try:
        # 测试 1: 初始化会话
        print("1. 初始化 MCP 会话...")
        print("-" * 70)
        init_result = await mcp_service.initialize()
        if "error" in init_result:
            print(f"❌ 初始化失败: {init_result['error']}")
            return
        print(f"✅ 初始化成功: {json.dumps(init_result, indent=2, ensure_ascii=False)}")
        print()
        
        # 测试 2: 列出所有工具
        print("2. 列出所有可用工具...")
        print("-" * 70)
        tools = await mcp_service.list_tools()
        if not tools:
            print("⚠️  未找到工具，可能是连接问题")
            return
        
        print(f"✅ 找到 {len(tools)} 个工具:")
        for i, tool in enumerate(tools, 1):
            # Tool 是 Pydantic 模型，使用属性访问而不是 .get()
            name = getattr(tool, 'name', 'Unknown')
            desc = getattr(tool, 'description', 'No description') or 'No description'
            print(f"  {i}. {name}")
            if desc:
                desc_short = desc[:80] + "..." if len(desc) > 80 else desc
                print(f"     描述: {desc_short}")
            
            # 显示参数信息
            input_schema = getattr(tool, 'inputSchema', None)
            if input_schema:
                properties = getattr(input_schema, 'properties', {}) or {}
                if properties:
                    print(f"     参数: {', '.join(properties.keys())}")
        print()
        
        # 测试 3: 调用 get_hot_kols 工具
        print("3. 测试调用工具: get_hot_kols")
        print("-" * 70)
        result = await mcp_service.call_tool(
            "get_hot_kols",
            {"limit": 5}
        )
        # CallToolResult 是 Pydantic 模型，需要转换为字典
        if hasattr(result, 'isError') and result.isError:
            print(f"❌ 工具调用失败: {getattr(result, 'content', 'Unknown error')}")
        else:
            print(f"✅ 工具调用成功")
            # 提取结构化内容或数据
            if hasattr(result, 'data') and result.data:
                print(f"结果: {json.dumps(result.data, indent=2, ensure_ascii=False, default=str)}")
            elif hasattr(result, 'structured_content') and result.structured_content:
                print(f"结果: {json.dumps(result.structured_content, indent=2, ensure_ascii=False, default=str)}")
            else:
                result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict() if hasattr(result, 'dict') else str(result)
                print(f"结果: {json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)}")
        print()
        
        # 测试 4: 调用 search_linkol_content 工具
        print("4. 测试调用工具: search_linkol_content")
        print("-" * 70)
        result = await mcp_service.call_tool(
            "search_linkol_content",
            {
                "query": "什么是 Linkol",
                "top_k": 3
            }
        )
        if hasattr(result, 'isError') and result.isError:
            print(f"❌ 工具调用失败: {getattr(result, 'content', 'Unknown error')}")
        else:
            print(f"✅ 工具调用成功")
            # 提取结构化内容或数据
            if hasattr(result, 'data') and result.data:
                print(f"结果: {json.dumps(result.data, indent=2, ensure_ascii=False, default=str)}")
            elif hasattr(result, 'structured_content') and result.structured_content:
                print(f"结果: {json.dumps(result.structured_content, indent=2, ensure_ascii=False, default=str)}")
            else:
                result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict() if hasattr(result, 'dict') else str(result)
                print(f"结果: {json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)}")
        print()
        
        # 测试 5: 调用 get_kol_price 工具
        print("5. 测试调用工具: get_kol_price")
        print("-" * 70)
        result = await mcp_service.call_tool(
            "get_kol_price",
            {"screen_name": "vis_eth"}
        )
        if hasattr(result, 'isError') and result.isError:
            print(f"❌ 工具调用失败: {getattr(result, 'content', 'Unknown error')}")
        else:
            print(f"✅ 工具调用成功")
            # 提取结构化内容或数据
            if hasattr(result, 'data') and result.data:
                print(f"结果: {json.dumps(result.data, indent=2, ensure_ascii=False, default=str)}")
            elif hasattr(result, 'structured_content') and result.structured_content:
                print(f"结果: {json.dumps(result.structured_content, indent=2, ensure_ascii=False, default=str)}")
            else:
                result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict() if hasattr(result, 'dict') else str(result)
                print(f"结果: {json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)}")
        print()
        
        print("=" * 70)
        print("✅ 所有测试完成！")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        print("\n提示: 请确保 MCP 服务器正在运行:")
        print("  docker compose up mcp")
        print("  或者")
        print("  poetry run python -m mcp.src.server")
        print("  或者")
        print("  ./mcp/run.sh")
    finally:
        # 关闭连接
        await mcp_service.close()


async def test_mcp_service_context_manager():
    """测试使用上下文管理器"""
    
    print("\n" + "=" * 70)
    print("测试 MCP 服务（使用上下文管理器）")
    print("=" * 70)
    print()
    
    try:
        async with MCPService() as mcp_service:
            # 列出工具
            tools = await mcp_service.list_tools()
            print(f"✅ 找到 {len(tools)} 个工具")
            
            # 调用工具
            result = await mcp_service.call_tool(
                "get_hot_kols",
                {"limit": 3}
            )
            if hasattr(result, 'isError') and result.isError:
                print(f"❌ 工具调用失败: {getattr(result, 'content', 'Unknown error')}")
            else:
                print(f"✅ 工具调用成功")
                result_dict = result.model_dump() if hasattr(result, 'model_dump') else result.dict() if hasattr(result, 'dict') else str(result)
                print(f"结果: {json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)}")
            
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_mcp_service())
    # 可选：测试上下文管理器
    # asyncio.run(test_mcp_service_context_manager())

