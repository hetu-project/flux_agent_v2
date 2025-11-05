FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml ./
COPY poetry.lock* ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    if [ -f poetry.lock ]; then \
        poetry install --no-root --no-interaction; \
    else \
        poetry install --no-root --no-interaction; \
    fi

# Copy source code
COPY src/ ./src/

# Expose port (if using HTTP transport)
EXPOSE 6001

# Run the MCP server (HTTP mode configured in code)
CMD ["python", "-m", "src.server"]

