"""
Integration tests for the text analysis pipeline of the RAG Chatbot for Robotics Book.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.services.rag_service import RAGService
from src.services.qdrant_service import QdrantService
from src.services.llm_service import LLMService
from src.models.query import QueryRequest


@pytest.fixture
def mock_qdrant_service():
    """Mock Qdrant service for testing."""
    service = Mock(spec=QdrantService)
    # Ensure that search_similar is not called when using selected text
    service.search_similar = AsyncMock(return_value=[])
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = Mock(spec=LLMService)
    service.generate_response.return_value = "Based on the selected text, robot kinematics deals with motion in robotic systems..."
    return service


@pytest.mark.asyncio
async def test_text_analysis_pipeline(mock_qdrant_service, mock_llm_service):
    """Test the complete text analysis pipeline with user-provided text."""
    # Create the RAG service with mocked dependencies
    rag_service = RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )
    
    # Prepare a query with selected text
    selected_text = "Robot kinematics is the study of motion in robotic systems..."
    query_request = QueryRequest(
        query="Explain this concept",
        selected_text=selected_text
    )
    
    # Process the query
    result = await rag_service.process_query(query_request)
    
    # Verify the result
    assert result.response_type == "using_selected_text"
    assert len(result.citations) == 0  # Should have no citations when using selected text
    assert "Based on the selected text" in result.content
    # Verify that Qdrant was not called since we're using selected text
    mock_qdrant_service.search_similar.assert_not_called()
    # Verify that the LLM service was called with the selected text
    mock_llm_service.generate_response.assert_called_once()
    # Check that the call included the selected text
    args, kwargs = mock_llm_service.generate_response.call_args
    assert selected_text in args or any(selected_text in str(v) for v in kwargs.values())


@pytest.mark.asyncio
async def test_text_analysis_pipeline_without_query(mock_qdrant_service, mock_llm_service):
    """Test the text analysis pipeline when no specific query is provided."""
    # Create the RAG service with mocked dependencies
    rag_service = RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )
    
    # Prepare a query with selected text but no specific query
    selected_text = "Robot kinematics is the study of motion in robotic systems..."
    query_request = QueryRequest(
        query="",  # Empty query
        selected_text=selected_text
    )
    
    # Process the query
    result = await rag_service.process_query(query_request)
    
    # Verify the result
    assert result.response_type == "using_selected_text"
    assert len(result.citations) == 0  # Should have no citations when using selected text
    # Verify that Qdrant was not called
    mock_qdrant_service.search_similar.assert_not_called()


@pytest.mark.asyncio
async def test_text_analysis_pipeline_with_llm_error(mock_qdrant_service, mock_llm_service):
    """Test the text analysis pipeline when LLM service fails."""
    # Update mock to raise an exception
    mock_llm_service.generate_response.side_effect = Exception("LLM API error")
    
    # Create the RAG service with mocked dependencies
    rag_service = RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )
    
    # Prepare a query with selected text
    selected_text = "Robot kinematics is the study of motion in robotic systems..."
    query_request = QueryRequest(
        query="Explain this concept",
        selected_text=selected_text
    )
    
    # Process the query
    result = await rag_service.process_query(query_request)
    
    # Verify the result is a fallback response
    assert result.response_type == "fallback_unavailable"
    assert len(result.citations) == 0
    assert "Unable to access book content at this moment" in result.content
    # Verify that Qdrant was not called
    mock_qdrant_service.search_similar.assert_not_called()


@pytest.mark.asyncio
async def test_process_selected_text_directly(mock_qdrant_service, mock_llm_service):
    """Test the direct process_selected_text method."""
    # Create the RAG service with mocked dependencies
    rag_service = RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )
    
    # Prepare a selected text
    selected_text = "Robot kinematics is the study of motion in robotic systems..."
    query = "Explain this concept"
    
    # Process the selected text directly
    result = await rag_service.process_selected_text(selected_text, query)
    
    # Verify the result
    assert result.response_type == "using_selected_text"
    assert len(result.citations) == 0  # Should have no citations when using selected text
    assert "Based on the selected text" in result.content or len(result.content) > 0
    # Verify that Qdrant was not called since we're using selected text
    mock_qdrant_service.search_similar.assert_not_called()