# MCP 服务器测试指南

这个文档说明如何测试 MCP 服务器。

## 快速开始

### 1. 启动 MCP 服务器

在项目根目录运行：

```bash
# 方式 1: 使用 Poetry
poetry run python -m mcp.src.server

# 方式 2: 使用启动脚本
./mcp/run.sh
```

服务器将在 `http://localhost:6001` 启动。

### 2. 运行测试脚本

在另一个终端窗口，运行测试脚本：

```bash
# 方式 1: 使用 Poetry
poetry run python mcp/test_client.py

# 方式 2: 直接运行
python mcp/test_client.py
```

## 测试脚本功能

测试脚本会依次测试以下功能：

1. **列出所有工具** - 查看 MCP 服务器提供的所有工具
2. **get_hot_kols** - 获取热门 KOL 列表
3. **search_linkol_content** - 搜索 Linkol 项目相关内容
4. **check_other_projects** - 检查用户问题中是否提到其他项目
5. **get_kol_price** - 获取指定 Twitter 用户的 KOL 估值价格

## 手动测试

你也可以使用 `curl` 或 `httpx` 手动测试：

```bash
# 列出所有工具
curl -X POST http://localhost:6001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }'

# 调用工具
curl -X POST http://localhost:6001/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "get_hot_kols",
      "arguments": {
        "limit": 5
      }
    }
  }'
```

## 注意事项

1. 确保 MCP 服务器正在运行（在端口 6001）
2. 确保已配置必要的环境变量（`.env` 文件）
3. 确保 Qdrant 服务正在运行（如果使用向量搜索功能）

## 故障排除

如果测试失败，请检查：

1. **服务器是否启动**: 查看服务器日志，确认没有错误
2. **端口是否被占用**: 确认端口 6001 没有被其他程序占用
3. **依赖是否安装**: 运行 `poetry install` 确保所有依赖已安装
4. **环境变量**: 检查 `.env` 文件中的配置是否正确

