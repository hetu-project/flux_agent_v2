#!/bin/bash
# MCP 服务器测试脚本 - 使用 curl

MCP_URL="http://localhost:6001"

echo "=========================================="
echo "测试 MCP 服务器"
echo "=========================================="
echo ""

# 测试 1: 列出所有工具 (tools/list)
echo "1. 测试 tools/list 方法..."
echo "----------------------------------------"
echo "尝试端点: $MCP_URL/mcp"
curl -X POST "$MCP_URL/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }' | jq '.' 2>/dev/null || echo "响应不是 JSON 格式或 jq 未安装"
echo ""
echo ""

# 测试 2: 尝试根路径
echo "2. 尝试根路径: $MCP_URL/"
echo "----------------------------------------"
curl -X POST "$MCP_URL/" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }' | jq '.' 2>/dev/null || echo "响应不是 JSON 格式或 jq 未安装"
echo ""
echo ""

# 测试 3: 尝试 /jsonrpc 端点
echo "3. 尝试 /jsonrpc 端点: $MCP_URL/jsonrpc"
echo "----------------------------------------"
curl -X POST "$MCP_URL/jsonrpc" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }' | jq '.' 2>/dev/null || echo "响应不是 JSON 格式或 jq 未安装"
echo ""
echo ""

# 测试 4: 尝试调用一个工具 (get_hot_kols)
echo "4. 测试调用工具: get_hot_kols"
echo "----------------------------------------"
curl -X POST "$MCP_URL/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "get_hot_kols",
      "arguments": {
        "limit": 5
      }
    }
  }' | jq '.' 2>/dev/null || echo "响应不是 JSON 格式或 jq 未安装"
echo ""
echo ""

# 测试 5: 检查服务器是否运行
echo "5. 检查服务器状态..."
echo "----------------------------------------"
if curl -s "$MCP_URL" > /dev/null 2>&1; then
  echo "✓ 服务器正在运行"
else
  echo "✗ 服务器未运行或无法访问"
  echo ""
  echo "提示: 启动 MCP 服务器:"
  echo "  poetry run python -m mcp.src.server"
  echo "  或"
  echo "  ./mcp/run.sh"
fi
echo ""

