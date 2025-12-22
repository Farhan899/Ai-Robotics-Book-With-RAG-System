"""
Integration tests for the RAG pipeline of the RAG Chatbot for Robotics Book.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.services.rag_service import RAGService
from src.services.qdrant_service import QdrantService
from src.services.llm_service import LLMService
from src.models.query import Query, QueryRequest
from src.models.document import RetrievedContext
from datetime import datetime


@pytest.fixture
def mock_qdrant_service():
    """Mock Qdrant service for testing."""
    service = Mock(spec=QdrantService)
    service.search_similar.return_value = [
        RetrievedContext(
            id="context1",
            content="Robot kinematics is the study of motion in robotic systems...",
            metadata={"chapter": "3", "section": "3.2", "source": "robotics_book"},
            similarity_score=0.85,
            source_document_id="doc1"
        )
    ]
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = Mock(spec=LLMService)
    service.generate_response.return_value = "Robot kinematics is the study of motion in robotic systems, focusing on the mathematical relationships that describe the position and movement of robotic systems."
    return service


@pytest.mark.asyncio
async def test_rag_pipeline_with_relevant_content(mock_qdrant_service, mock_llm_service):
    """Test the complete RAG pipeline when relevant content is found."""
    # Create the RAG service with mocked dependencies
    rag_service = RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )
    
    # Prepare a query
    query_request = QueryRequest(
        query="What are the fundamentals of robot kinematics?"
    )
    
    # Process the query
    result = await rag_service.process_query(query_request)
    
    # Verify the result
    assert result.response_type == "from_book"
    assert len(result.citations) > 0
    assert "Book Section:" in result.citations[0]
    assert len(result.content) > 0


@pytest.mark.asyncio
async def test_rag_pipeline_with_no_relevant_content(mock_qdrant_service, mock_llm_service):
    """Test the RAG pipeline when no relevant content is found."""
    # Update mock to return empty results
    mock_qdrant_service.search_similar.return_value = []
    
    # Create the RAG service with mocked dependencies
    rag_service = RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )
    
    # Prepare a query
    query_request = QueryRequest(
        query="What are the fundamentals of robot kinematics?"
    )
    
    # Process the query
    result = await rag_service.process_query(query_request)
    
    # Verify the result
    assert result.response_type == "fallback_general"
    assert len(result.citations) == 0
    assert len(result.content) > 0


@pytest.mark.asyncio
async def test_rag_pipeline_with_qdrant_failure(mock_qdrant_service, mock_llm_service):
    """Test the RAG pipeline when Qdrant service fails."""
    # Update mock to raise an exception
    mock_qdrant_service.search_similar.side_effect = Exception("Qdrant connection failed")
    
    # Create the RAG service with mocked dependencies
    rag_service = RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )
    
    # Prepare a query
    query_request = QueryRequest(
        query="What are the fundamentals of robot kinematics?"
    )
    
    # Process the query
    result = await rag_service.process_query(query_request)
    
    # Verify the result is a fallback response
    assert result.response_type == "fallback_unavailable"
    assert len(result.citations) == 0
    assert "Unable to access book content at this moment" in result.content