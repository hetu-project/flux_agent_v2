"""Logging configuration with hierarchical loggers."""

import logging
import sys
from typing import Optional
from pathlib import Path
from src.config import get_settings

settings = get_settings()


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Setup hierarchical logging configuration.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        log_format: Optional custom log format
    """
    level = log_level or settings.log_level
    log_file_path = log_file or settings.log_file
    log_format = log_format or (
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    
    # Create logs directory if logging to file
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # File logs everything
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific log levels for different modules
    _configure_module_loggers()


def _configure_module_loggers():
    """Configure log levels for different modules."""
    # Core modules - INFO level
    logging.getLogger("src.agents").setLevel(logging.INFO)
    logging.getLogger("src.services").setLevel(logging.INFO)
    logging.getLogger("src.repositories").setLevel(logging.INFO)
    
    # API layer - INFO level
    logging.getLogger("src.api").setLevel(logging.INFO)
    
    # External libraries - WARNING level (reduce noise)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
        
    Examples:
        logger = get_logger(__name__)
        logger.info("This is an info message")
    """
    return logging.getLogger(name)


# Initialize logging on module import
setup_logging()

