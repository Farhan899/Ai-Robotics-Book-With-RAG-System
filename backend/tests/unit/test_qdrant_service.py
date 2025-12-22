"""
Unit tests for the Qdrant Service of the RAG Chatbot for Robotics Book.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.services.qdrant_service import QdrantService
from src.models.document import RetrievedContext


@pytest.fixture
def mock_async_client():
    """Mock AsyncQdrantClient for testing."""
    client = Mock(spec=AsyncQdrantClient)
    return client


@pytest.fixture
def qdrant_service(mock_async_client):
    """Create a QdrantService instance with mocked dependencies."""
    service = QdrantService(
        client=mock_async_client,
        collection_name="robotics_docs"
    )
    return service


@pytest.mark.asyncio
async def test_search_similar(qdrant_service, mock_async_client):
    """Test searching for similar content in Qdrant."""
    # Arrange
    query_text = "What are the fundamentals of robot kinematics?"
    mock_search_result = [
        models.ScoredPoint(
            id="doc1",
            version=1,
            score=0.85,
            payload={
                "content": "Robot kinematics is the study of motion in robotic systems...",
                "metadata": {"chapter": "3", "section": "3.2", "source": "robotics_book"}
            },
            vector=[0.1, 0.2, 0.3]  # Mock vector
        )
    ]
    mock_async_client.search = AsyncMock(return_value=mock_search_result)
    
    # Act
    results = await qdrant_service.search_similar(query_text, limit=5)
    
    # Assert
    mock_async_client.search.assert_called_once_with(
        collection_name="robotics_docs",
        query_text=query_text,
        limit=5,
        with_payload=True,
        with_vectors=False
    )
    assert len(results) == 1
    assert isinstance(results[0], RetrievedContext)
    assert results[0].content == "Robot kinematics is the study of motion in robotic systems..."
    assert results[0].metadata["chapter"] == "3"
    assert results[0].similarity_score == 0.85


@pytest.mark.asyncio
async def test_search_similar_no_results(qdrant_service, mock_async_client):
    """Test searching for similar content when no results are found."""
    # Arrange
    query_text = "What are the fundamentals of robot kinematics?"
    mock_async_client.search = AsyncMock(return_value=[])
    
    # Act
    results = await qdrant_service.search_similar(query_text, limit=5)
    
    # Assert
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_similar_exception(qdrant_service, mock_async_client):
    """Test searching when Qdrant client raises an exception."""
    # Arrange
    query_text = "What are the fundamentals of robot kinematics?"
    mock_async_client.search = AsyncMock(side_effect=Exception("Connection failed"))
    
    # Act & Assert
    with pytest.raises(Exception):
        await qdrant_service.search_similar(query_text, limit=5)


@pytest.mark.asyncio
async def test_search_similar_with_threshold(qdrant_service, mock_async_client):
    """Test searching for similar content with a minimum similarity threshold."""
    # Arrange
    query_text = "What are the fundamentals of robot kinematics?"
    mock_search_result = [
        models.ScoredPoint(
            id="doc1",
            version=1,
            score=0.3,  # Below threshold
            payload={
                "content": "This is loosely related content...",
                "metadata": {"chapter": "10", "section": "10.1", "source": "robotics_book"}
            },
            vector=[0.1, 0.2, 0.3]
        ),
        models.ScoredPoint(
            id="doc2",
            version=1,
            score=0.85,  # Above threshold
            payload={
                "content": "Robot kinematics is the study of motion in robotic systems...",
                "metadata": {"chapter": "3", "section": "3.2", "source": "robotics_book"}
            },
            vector=[0.4, 0.5, 0.6]
        )
    ]
    mock_async_client.search = AsyncMock(return_value=mock_search_result)
    
    # Act
    results = await qdrant_service.search_similar(query_text, limit=5)
    
    # Assert
    assert len(results) == 2  # Both results should be included since we're not filtering by threshold in the service
    # The actual filtering by threshold would be done in RAG service logic
    assert results[1].similarity_score == 0.85


@pytest.mark.asyncio
async def test_init_with_custom_collection_name():
    """Test initialization with a custom collection name."""
    # Arrange
    mock_client = Mock(spec=AsyncQdrantClient)
    
    # Act
    service = QdrantService(
        client=mock_client,
        collection_name="custom_collection"
    )
    
    # Assert
    assert service.collection_name == "custom_collection"