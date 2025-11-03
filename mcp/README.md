# MCP Server (使用 FastMCP 框架)

独立的 Model Context Protocol 服务器，使用 FastMCP 框架构建，提供工具供 FastAPI 客户端调用。

## 架构

```
┌──────────────┐         MCP Protocol         ┌──────────────┐
│  FastAPI     │  ───────────────────────────► │  MCP Server  │
│  (客户端)    │  (调用工具)                    │  (FastMCP)   │
└──────────────┘                                └──────────────┘
```

## 目录结构（FastMCP 最佳实践）

```
mcp/
├── src/
│   ├── __init__.py
│   ├── server.py              # MCP 服务器主文件（使用 FastMCP）
│   │
│   ├── core/                   # 核心服务管理
│   │   ├── __init__.py
│   │   └── services.py         # 服务初始化和依赖注入
│   │
│   ├── tools/                  # 工具实现
│   │   ├── __init__.py
│   │   └── linkol_tools.py     # Linkol 工具实现
│   │
│   ├── services/               # 服务层（从主项目复制）
│   ├── repositories/           # 仓库层（从主项目复制）
│   ├── models/                 # 数据模型（从主项目复制）
│   ├── utils/                  # 工具函数（从主项目复制）
│   └── config.py               # 配置（从主项目复制）
│
├── run.sh                      # 启动脚本
└── README.md
```

## 架构设计

### 1. **server.py** - MCP 服务器入口
- 使用 FastMCP 框架初始化
- 使用 `@mcp.tool()` 装饰器注册工具
- 工具函数直接调用工具实现

### 2. **core/services.py** - 服务管理
- 统一的服务初始化（单例模式）
- 服务依赖注入
- 延迟初始化，按需创建

### 3. **tools/linkol_tools.py** - 工具实现
- 分离工具逻辑和注册
- 便于测试和维护
- 可复用的实现函数

## 功能

MCP 服务器提供以下工具（使用 `@mcp.tool()` 装饰器定义）：

1. **get_kol_price** - 获取指定 Twitter 用户的 KOL 估值价格
2. **get_hot_kols** - 获取热门 KOL 列表
3. **search_linkol_content** - 在 Linkol 项目知识库中搜索相关内容
4. **check_other_projects** - 检查用户问题中是否提到其他项目

## 安装

### 1. 安装依赖

```bash
# 在项目根目录
poetry install
```

这会自动安装 `fastmcp` 依赖。

### 2. 验证安装

```bash
poetry run python -c "import fastmcp; print('FastMCP installed')"
```

## 运行

### 方式 1: 直接运行（推荐）

```bash
# 在项目根目录
poetry run python -m mcp.src.server
```

### 方式 2: 使用脚本

```bash
./mcp/run.sh
```

### 方式 3: 使用 Python 直接运行

```bash
cd ..
python -m mcp.src.server
```

## FastMCP 使用模式

### 工具定义模式

```python
# server.py
from fastmcp import FastMCP
from tools.linkol_tools import get_kol_price_impl

mcp = FastMCP("Hetu Agent MCP Server")

@mcp.tool()
async def get_kol_price(screen_name: str) -> Dict[str, Any]:
    """获取指定Twitter用户的KOL估值价格"""
    return await get_kol_price_impl(screen_name)

if __name__ == "__main__":
    mcp.run()
```

### 服务管理模式

```python
# core/services.py
_qdrant_service: Optional[QdrantService] = None

def init_services():
    """初始化所有服务（单例模式）"""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service

def get_qdrant_service() -> QdrantService:
    """获取服务实例"""
    if _qdrant_service is None:
        init_services()
    return _qdrant_service
```

## 配置

MCP 服务器使用主项目的配置文件（`.env`），需要以下配置：

```env
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Linkol API
LINKOL_URL=https://api.linkol.com
LINKOL_API_KEY=your_api_key

# Embedding
AIHUBMIX_API_KEY=your_key
AIHUBMIX_BASE_URL=https://api.aihubmix.com
EMBEDDING_MODEL=text-embedding-ada-002
```

## 添加新工具

使用 FastMCP，添加新工具非常简单：

### 步骤 1: 在 `tools/` 目录下添加实现

```python
# tools/my_new_tools.py
async def my_new_tool_impl(param1: str, param2: int = 10) -> Dict[str, Any]:
    """工具实现逻辑"""
    # 实现代码
    return {"result": "success"}
```

### 步骤 2: 在 `server.py` 中注册工具

```python
# server.py
from tools.my_new_tools import my_new_tool_impl

@mcp.tool()
async def my_new_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    我的新工具
    
    Args:
        param1: 参数1的说明
        param2: 参数2的说明，默认10
    
    Returns:
        返回结果的说明
    """
    return await my_new_tool_impl(param1, param2)
```

FastMCP 会自动：
- 根据类型注解生成参数 Schema
- 根据文档字符串生成工具描述
- 注册工具到 MCP 服务器

## 优势

使用 FastMCP 框架的优势：

- ✅ **简洁**: 使用装饰器即可定义工具，无需手动注册
- ✅ **自动**: 自动处理工具描述、参数验证、调用路由
- ✅ **类型安全**: 利用 Python 类型注解自动生成 Schema
- ✅ **标准化**: 遵循 MCP 协议标准
- ✅ **分层清晰**: 工具注册、实现、服务管理分离

## 开发建议

1. **保持工具实现独立**: 工具实现放在 `tools/` 目录，便于测试
2. **使用服务管理器**: 通过 `core/services.py` 统一管理服务依赖
3. **清晰的文档字符串**: FastMCP 使用文档字符串生成工具描述
4. **类型注解**: 充分利用类型注解，FastMCP 会自动生成 Schema
