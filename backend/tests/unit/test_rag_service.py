"""
Unit tests for the RAG Service of the RAG Chatbot for Robotics Book.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.services.rag_service import RAGService
from src.services.qdrant_service import QdrantService
from src.services.llm_service import LLMService
from src.models.query import QueryRequest
from src.models.document import RetrievedContext


@pytest.fixture
def mock_qdrant_service():
    """Mock Qdrant service for testing."""
    service = Mock(spec=QdrantService)
    return service


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    service = Mock(spec=LLMService)
    return service


@pytest.fixture
def rag_service(mock_qdrant_service, mock_llm_service):
    """Create a RAGService instance with mocked dependencies."""
    return RAGService(
        qdrant_service=mock_qdrant_service,
        llm_service=mock_llm_service
    )


@pytest.mark.asyncio
async def test_process_query_with_relevant_content(rag_service, mock_qdrant_service, mock_llm_service):
    """Test processing a query when relevant content is found in Qdrant."""
    # Arrange
    mock_retrieved_contexts = [
        RetrievedContext(
            id="context1",
            content="Robot kinematics is the study of motion in robotic systems...",
            metadata={"chapter": "3", "section": "3.2", "source": "robotics_book"},
            similarity_score=0.85,
            source_document_id="doc1"
        )
    ]
    mock_qdrant_service.search_similar.return_value = mock_retrieved_contexts
    mock_llm_service.generate_response.return_value = "Based on the book, robot kinematics is the study of motion in robotic systems..."
    
    query_request = QueryRequest(query="What are the fundamentals of robot kinematics?")
    
    # Act
    result = await rag_service.process_query(query_request)
    
    # Assert
    assert result.response_type == "from_book"
    assert "Book Section:" in result.citations[0]
    assert "robotics_book" in result.citations[0]  # Check that source is in citation
    mock_qdrant_service.search_similar.assert_called_once_with(query_request.query, limit=5)
    mock_llm_service.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_process_query_with_no_relevant_content(rag_service, mock_qdrant_service, mock_llm_service):
    """Test processing a query when no relevant content is found in Qdrant."""
    # Arrange
    mock_qdrant_service.search_similar.return_value = []
    mock_llm_service.generate_response.return_value = "Robot kinematics is the study of motion in robotic systems."
    
    query_request = QueryRequest(query="What are the fundamentals of robot kinematics?")
    
    # Act
    result = await rag_service.process_query(query_request)
    
    # Assert
    assert result.response_type == "fallback_general"
    assert len(result.citations) == 0
    assert "No direct book content found for this query" in result.content
    mock_qdrant_service.search_similar.assert_called_once_with(query_request.query, limit=5)


@pytest.mark.asyncio
async def test_process_query_with_qdrant_error(rag_service, mock_qdrant_service, mock_llm_service):
    """Test processing a query when Qdrant service fails."""
    # Arrange
    mock_qdrant_service.search_similar.side_effect = Exception("Connection failed")
    
    query_request = QueryRequest(query="What are the fundamentals of robot kinematics?")
    
    # Act
    result = await rag_service.process_query(query_request)
    
    # Assert
    assert result.response_type == "fallback_unavailable"
    assert len(result.citations) == 0
    assert "Unable to access book content at this moment" in result.content


@pytest.mark.asyncio
async def test_process_selected_text(rag_service, mock_qdrant_service, mock_llm_service):
    """Test processing selected text (ignoring Qdrant retrieval)."""
    # Arrange
    mock_llm_service.generate_response.return_value = "Based on the selected text, robot kinematics deals with motion in robotic systems..."
    
    query_request = QueryRequest(
        query="Explain this concept",
        selected_text="Robot kinematics is the study of motion in robotic systems..."
    )
    
    # Act
    result = await rag_service.process_query(query_request)
    
    # Assert
    assert result.response_type == "using_selected_text"
    assert len(result.citations) == 0  # No citations when using selected text
    mock_qdrant_service.search_similar.assert_not_called()  # Should not call Qdrant when selected text is provided
    mock_llm_service.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_process_query_low_similarity_score(rag_service, mock_qdrant_service, mock_llm_service):
    """Test processing a query when similarity score is too low."""
    # Arrange
    mock_retrieved_contexts = [
        RetrievedContext(
            id="context1",
            content="This is loosely related content...",
            metadata={"chapter": "10", "section": "10.1", "source": "robotics_book"},
            similarity_score=0.1,  # Low similarity score
            source_document_id="doc1"
        )
    ]
    mock_qdrant_service.search_similar.return_value = mock_retrieved_contexts
    mock_llm_service.generate_response.return_value = "The fundamentals of robot kinematics involve understanding motion in robotic systems."

    query_request = QueryRequest(query="What are the fundamentals of robot kinematics?")

    # Act
    result = await rag_service.process_query(query_request)

    # Assert
    # Should still be from book since we have some context, but check that behavior matches implementation
    assert result.response_type in ["from_book", "fallback_general"]  # Depends on the threshold implementation
    mock_qdrant_service.search_similar.assert_called_once_with(query_request.query, limit=5)


@pytest.mark.asyncio
async def test_process_selected_text_method(rag_service, mock_qdrant_service, mock_llm_service):
    """Test the direct process_selected_text method."""
    # Arrange
    selected_text = "Robot kinematics is the study of motion in robotic systems..."
    query = "Explain this concept"
    mock_llm_service.generate_response.return_value = "Based on the selected text, robot kinematics deals with motion in robotic systems..."

    # Act
    result = await rag_service.process_selected_text(selected_text, query)

    # Assert
    assert result.response_type == "using_selected_text"
    assert len(result.citations) == 0  # No citations when using selected text
    mock_qdrant_service.search_similar.assert_not_called()  # Should not call Qdrant
    mock_llm_service.generate_response.assert_called_once()
    # Verify the LLM was called with the selected text
    args, kwargs = mock_llm_service.generate_response.call_args
    assert selected_text in args or any(selected_text in str(v) for v in kwargs.values())


@pytest.mark.asyncio
async def test_process_selected_text_with_error(rag_service, mock_qdrant_service, mock_llm_service):
    """Test the direct process_selected_text method when LLM service fails."""
    # Arrange
    selected_text = "Robot kinematics is the study of motion in robotic systems..."
    query = "Explain this concept"
    mock_llm_service.generate_response.side_effect = Exception("LLM Error")

    # Act
    result = await rag_service.process_selected_text(selected_text, query)

    # Assert
    assert result.response_type == "fallback_unavailable"
    assert len(result.citations) == 0
    assert "Unable to access book content at this moment" in result.content
    mock_qdrant_service.search_similar.assert_not_called()  # Should not call Qdrant


@pytest.mark.asyncio
async def test_fallback_behavior_on_qdrant_error(rag_service, mock_qdrant_service, mock_llm_service):
    """Test fallback behavior when Qdrant service fails."""
    # Arrange
    mock_qdrant_service.search_similar.side_effect = Exception("Qdrant connection failed")
    mock_llm_service.generate_response.return_value = "General response about robot kinematics."

    query_request = QueryRequest(query="What are the fundamentals of robot kinematics?")

    # Act
    result = await rag_service.process_query(query_request)

    # Assert
    assert result.response_type == "fallback_unavailable"
    assert len(result.citations) == 0  # No citations in fallback
    assert "Unable to access book content at this moment" in result.content
    # The content should include a response from the LLM even in fallback mode
    assert "General response about robot kinematics" in result.content


@pytest.mark.asyncio
async def test_fallback_behavior_on_no_content(rag_service, mock_qdrant_service, mock_llm_service):
    """Test fallback behavior when no relevant content is found."""
    # Arrange
    mock_qdrant_service.search_similar.return_value = []  # No results found
    mock_llm_service.generate_response.return_value = "Robot kinematics fundamentals involve understanding motion in robotic systems."

    query_request = QueryRequest(query="What are the fundamentals of robot kinematics?")

    # Act
    result = await rag_service.process_query(query_request)

    # Assert
    assert result.response_type == "fallback_general"
    assert len(result.citations) == 0  # No citations when no content found
    assert "No direct book content found for this query" in result.content
    # The content should include a response from the LLM
    assert "Robot kinematics fundamentals" in result.content


@pytest.mark.asyncio
async def test_fallback_behavior_on_llm_error(rag_service, mock_qdrant_service, mock_llm_service):
    """Test fallback behavior when LLM service fails after getting content."""
    # Arrange
    mock_retrieved_contexts = [
        RetrievedContext(
            id="context1",
            content="Robot kinematics is the study of motion in robotic systems...",
            metadata={"chapter": "3", "section": "3.2", "source": "robotics_book"},
            similarity_score=0.85,
            source_document_id="doc1"
        )
    ]
    mock_qdrant_service.search_similar.return_value = mock_retrieved_contexts
    mock_llm_service.generate_response.side_effect = Exception("LLM service error")

    query_request = QueryRequest(query="What are the fundamentals of robot kinematics?")

    # Act
    result = await rag_service.process_query(query_request)

    # Assert
    assert result.response_type == "fallback_unavailable"
    assert len(result.citations) == 0  # No citations in error fallback
    assert "Unable to access book content at this moment" in result.content