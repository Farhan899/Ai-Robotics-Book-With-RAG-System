"""
Response model for the RAG Chatbot for Robotics Book.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ResponseType(str, Enum):
    """
    Enum representing the type of response generated.
    """
    from_book = "from_book"
    fallback_general = "fallback_general"
    fallback_unavailable = "fallback_unavailable"
    using_selected_text = "using_selected_text"


class ChatResponse(BaseModel):
    """
    Model representing the system's response to the user's query.
    """
    id: str
    content: str
    citations: List[str] = []
    response_type: ResponseType
    timestamp: datetime


class ErrorResponse(BaseModel):
    """
    Model representing an error response.
    """
    error: str
    message: str