"""
Query model for the RAG Chatbot for Robotics Book.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """
    Model representing a user's query to the system.
    """
    query: str = Field(
        ..., 
        description="The question or query from the user",
        min_length=1, 
        max_length=2000
    )
    selected_text: Optional[str] = Field(
        None,
        description="Optional specific text from the book for focused analysis",
        max_length=5000
    )


class Query(BaseModel):
    """
    Model representing a user's input to the system with additional metadata.
    """
    id: str
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=2000
    )
    timestamp: datetime
    user_id: Optional[str] = None
    selected_text: Optional[str] = Field(
        None,
        max_length=5000
    )