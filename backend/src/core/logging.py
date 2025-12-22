"""
Logging infrastructure for the RAG Chatbot for Robotics Book.
"""
import logging
import sys
from logging import config
from typing import Dict, Any


def setup_logging() -> None:
    """
    Set up logging configuration for the application.
    """
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(asctime)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(asctime)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
            "rag_chatbot": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    config.dictConfig(logging_config)


# Set up logging
setup_logging()

# Create logger for the application
logger = logging.getLogger("rag_chatbot")