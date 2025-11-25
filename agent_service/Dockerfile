FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (align with local dev)
RUN pip install "poetry==2.1.3"

# Copy poetry files (poetry.lock is optional)
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

# Copy alembic configuration and migrations
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Expose port
EXPOSE 8000

# Run the application with reload for development
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

