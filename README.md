# Hetu Agent

RAG Agent for Twitter project analysis using Qdrant vector database.

## Tech Stack

- **Python**: 3.11.11
- **Backend**: FastAPI
- **Vector DB**: Qdrant
- **LLM**: OpenAI / AIHubMix (可切换)
- **Package Manager**: Poetry

## Features

- Twitter data collection via API
- Vector embedding and storage in Qdrant
- RAG-based information retrieval
- Conversational agent for project analysis
- **支持多种LLM提供商**：OpenAI、AIHubMix

## Setup

### 1. Install Dependencies

```bash
# Install poetry if not already installed
pip install poetry

# Install project dependencies
poetry install
```

### 2. Start Qdrant

```bash
# Start Qdrant in Docker
docker compose up -d
```

Qdrant will be available at `http://localhost:6333`

### 3. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your favorite editor
```

#### Configuration Options

**使用 OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-ada-002
```

**使用 AIHubMix (更便宜):**
```env
LLM_PROVIDER=aihubmix
AIHUBMIX_API_KEY=your_aihubmix_api_key
AIHUBMIX_BASE_URL=https://aihubmix.com
# AIHubMix支持多种模型，你可以根据文档选择合适的chat_model
CHAT_MODEL=claude-3-5-sonnet  # 或其他AIHubMix支持的模型
EMBEDDING_MODEL=text-embedding-ada-002  # 或AIHubMix的embedding模型
```

**其他必要配置:**
```env
# Twitter API
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 4. Run the Application

```bash
# Navigate to agent_service directory
cd agent_service

# Start FastAPI app
poetry run uvicorn src.main:app --reload

# Or use poetry shell
poetry shell
uvicorn src.main:app --reload
```

API will be available at:
- http://localhost:8000
- API Docs: http://localhost:8000/docs

## Usage

### 1. Collect Tweets

```bash
curl -X POST http://localhost:8000/api/tweets/collect \
  -H "Content-Type: application/json" \
  -d '{
    "project_name": "langchain",
    "username": "langchain_ai",
    "max_tweets": 100
  }'
```

### 2. Chat with Agent

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the latest update about LangChain?",
    "project": "langchain"
  }'
```

### 3. Check Collection Info

```bash
curl http://localhost:8000/api/collections/twitter_tweets/info
```

## Project Structure

```
hetu-agent/
├── agent_service/    # Agent service (microservice)
│   ├── src/          # Source code
│   │   ├── main.py   # FastAPI application
│   │   ├── config.py # Configuration
│   │   ├── models/   # Data models
│   │   ├── services/ # Services
│   │   └── agents/   # Agents
│   ├── pyproject.toml # Poetry dependencies
│   └── Dockerfile     # Docker configuration
├── mcp/              # MCP service
├── scripts/          # Utility scripts
├── docker-compose.yml # Docker compose setup
└── README.md         # This file
```

## LLM Provider Switching

项目支持在OpenAI和AIHubMix之间切换，只需修改`.env`文件中的`LLM_PROVIDER`设置即可。

**费用对比:**
- OpenAI: 官方定价
- AIHubMix: 通常更便宜，支持多种模型(Claude, Gemini, Qwen等)

## API Endpoints

### GET /
Root endpoint

### POST /api/tweets/collect
Collect tweets from Twitter and store in Qdrant

**Request Body:**
```json
{
  "project_name": "string",
  "username": "string (optional)",
  "max_tweets": 100,
  "query": "string (optional)"
}
```

### POST /api/chat
Chat with the RAG agent

**Request Body:**
```json
{
  "query": "your question",
  "project": "project_name (optional)"
}
```

### GET /api/collections/{collection_name}/info
Get collection information

## Development

### Code Quality

```bash
# Navigate to agent_service directory
cd agent_service

# Format code
poetry run black src/

# Lint code
poetry run ruff check src/

# Run tests
poetry run pytest
```

## Troubleshooting

### Qdrant connection error

Make sure Qdrant is running:
```bash
curl http://localhost:6333
```

### OpenAI/AIHubMix API error

Check your API key in `.env` file and verify the `LLM_PROVIDER` setting

### Twitter API error

Ensure you have a valid bearer token from Twitter Developer Portal

## License

MIT
