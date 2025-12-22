"""
Document model for the RAG Chatbot for Robotics Book.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class Document(BaseModel):
    """
    Model representing a document in the Qdrant vector database.
    """
    id: str
    content: str
    metadata: Dict[str, str]  # Metadata including source, section, chapter, etc.
    embedding: Optional[List[float]] = None  # The vector embedding of the content
    created_at: datetime


class RetrievedContext(BaseModel):
    """
    Model representing the context retrieved from the Qdrant database.
    """
    id: str
    content: str
    metadata: Dict[str, str]  # Additional metadata about the source (e.g., chapter, page, section)
    similarity_score: float  # The similarity score from the vector search
    source_document_id: str  # Identifier for the original document