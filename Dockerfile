FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry files (poetry.lock is optional)
COPY pyproject.toml ./
COPY poetry.lock* ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false && \
    if [ -f poetry.lock ]; then \
        poetry install --no-root --no-interaction; \
    else \
        poetry install --no-root --no-interaction --no-lock; \
    fi

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Expose port
EXPOSE 8000

# Run the application with reload for development
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

