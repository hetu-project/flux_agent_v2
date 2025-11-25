"""Script to initialize agents in the database."""

import asyncio
import sys
import os

# Add the app directory to the path so we can import from src
script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(script_dir))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from sqlalchemy import select
from src.services.database import get_async_session
from src.sql_models.agent import Agent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Agent definitions with name and description
AGENTS = [
    {
        "name": "company",
        "description": "陪伴助手，提供情感陪伴、日常对话和情绪支持。温暖、友善、善解人意，像朋友一样交流。",
        "is_active": True,
    },
    {
        "name": "fortune",
        "description": "占卜预测助手，根据姓名、出生年份和星座预测明日运势。支持整体运势、桃花运、事业运、财运等多种运势类型。",
        "is_active": True,
    },
    {
        "name": "bazi",
        "description": "八字算命助手，根据农历出生日期、时间和地点计算四柱八字，分析命理特点、性格特征和运势建议。",
        "is_active": True,
    },
    {
        "name": "tarot",
        "description": "塔罗牌占卜助手，为用户的问题提供塔罗牌占卜和指导。使用标准78张塔罗牌进行占卜。",
        "is_active": True,
    },
    {
        "name": "crypto",
        "description": "加密货币助手，提供加密货币相关的信息、分析和建议。",
        "is_active": True,
    },
    {
        "name": "health",
        "description": "健康助手，提供健康相关的咨询和建议。",
        "is_active": True,
    },
    {
        "name": "rag",
        "description": "RAG（检索增强生成）助手，基于项目内容进行问答，支持文档检索和上下文理解。",
        "is_active": True,
    },
    {
        "name": "rag_v2",
        "description": "RAG助手V2版本，增强的检索增强生成能力，支持更精准的文档检索和回答。",
        "is_active": True,
    },
    {
        "name": "linkol",
        "description": "Linkol助手，提供Linkol项目相关的信息和服务。",
        "is_active": True,
    },
    {
        "name": "linkol_v2",
        "description": "Linkol助手V2版本，增强的Linkol项目服务能力。",
        "is_active": True,
    },
    {
        "name": "hetu",
        "description": "Hetu Protocol助手，提供Hetu协议相关的信息、分析和问答服务。",
        "is_active": True,
    },
    {
        "name": "hetu_v2",
        "description": "Hetu Protocol助手V2版本，增强的Hetu协议服务能力。",
        "is_active": True,
    },
    {
        "name": "hetu_mcp",
        "description": "Hetu MCP助手，基于MCP（Model Context Protocol）的Hetu协议助手，支持工具调用。",
        "is_active": True,
    },
    {
        "name": "mcp",
        "description": "MCP助手，基于MCP（Model Context Protocol）的通用助手，支持工具调用和扩展功能。",
        "is_active": True,
    },
]


async def init_agents():
    """Initialize agents in the database."""
    print("=" * 80)
    print("Initializing Agents in Database")
    print("=" * 80)
    
    async with get_async_session() as session:
        created_count = 0
        updated_count = 0
        skipped_count = 0
        
        for agent_data in AGENTS:
            name = agent_data["name"]
            description = agent_data["description"]
            is_active = agent_data.get("is_active", True)
            
            try:
                # Check if agent already exists
                stmt = select(Agent).where(Agent.name == name)
                result = await session.execute(stmt)
                existing_agent = result.scalar_one_or_none()
                
                if existing_agent:
                    # Update existing agent
                    existing_agent.description = description
                    existing_agent.is_active = is_active
                    updated_count += 1
                    print(f"✅ Updated agent: {name}")
                    logger.info(f"Updated agent: {name}")
                else:
                    # Create new agent
                    new_agent = Agent(
                        name=name,
                        description=description,
                        is_active=is_active
                    )
                    session.add(new_agent)
                    created_count += 1
                    print(f"✅ Created agent: {name}")
                    logger.info(f"Created agent: {name}")
                
            except Exception as e:
                skipped_count += 1
                print(f"❌ Error processing agent '{name}': {e}")
                logger.error(f"Error processing agent '{name}': {e}", exc_info=True)
        
        # Commit all changes
        try:
            await session.commit()
            print("\n" + "=" * 80)
            print("Summary:")
            print(f"  Created: {created_count} agents")
            print(f"  Updated: {updated_count} agents")
            print(f"  Skipped: {skipped_count} agents")
            print(f"  Total: {len(AGENTS)} agents")
            print("=" * 80)
            logger.info(f"Agent initialization completed: {created_count} created, {updated_count} updated, {skipped_count} skipped")
        except Exception as e:
            await session.rollback()
            print(f"\n❌ Error committing changes: {e}")
            logger.error(f"Error committing changes: {e}", exc_info=True)
            raise


async def list_agents():
    """List all agents in the database."""
    print("=" * 80)
    print("Listing All Agents in Database")
    print("=" * 80)
    
    async with get_async_session() as session:
        stmt = select(Agent).order_by(Agent.name)
        result = await session.execute(stmt)
        agents = result.scalars().all()
        
        if not agents:
            print("No agents found in database.")
            return
        
        print(f"\nFound {len(agents)} agents:\n")
        for agent in agents:
            status = "✅ Active" if agent.is_active else "❌ Inactive"
            print(f"  [{status}] {agent.name} (ID: {agent.id})")
            if agent.description:
                # Truncate description if too long
                desc = agent.description[:80] + "..." if len(agent.description) > 80 else agent.description
                print(f"      {desc}")
            print()


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize or list agents in the database")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all agents in the database instead of initializing"
    )
    args = parser.parse_args()
    
    if args.list:
        await list_agents()
    else:
        await init_agents()
        print("\n💡 Tip: Use --list flag to view all agents in the database")


if __name__ == "__main__":
    asyncio.run(main())

