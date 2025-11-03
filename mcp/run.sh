#!/bin/bash
# MCP Server 启动脚本

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 使用 Poetry 运行（如果可用）
if command -v poetry &> /dev/null; then
    poetry run python -m mcp.src.server
else
    # 直接运行（需要确保依赖已安装）
    python -m mcp.src.server
fi

