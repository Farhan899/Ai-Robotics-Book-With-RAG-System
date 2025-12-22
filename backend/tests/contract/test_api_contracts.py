"""
Contract tests for the API endpoints of the RAG Chatbot for Robotics Book.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.api.main import app

client = TestClient(app)


def test_chat_endpoint_contract():
    """
    Test the contract for the /chat endpoint.
    """
    # Mock the services to test only the API contract
    with patch('src.services.rag_service.RAGService') as mock_rag_service:
        # Mock the response
        mock_response = Mock()
        mock_response.id = "test_response_id"
        mock_response.content = "This is a test response"
        mock_response.citations = ["Book Section: Chapter 1 - Introduction"]
        mock_response.response_type = "from_book"
        mock_response.timestamp = "2023-10-20T10:00:00Z"
        
        mock_rag_service_instance = Mock()
        mock_rag_service_instance.process_query.return_value = mock_response
        mock_rag_service.return_value = mock_rag_service_instance
        
        # Make a request to the endpoint
        response = client.post(
            "/chat",
            json={"query": "What are the fundamentals of robotics?"},
            headers={"X-API-Key": "test-key"}
        )
        
        # Verify the response status
        assert response.status_code == 200
        
        # Verify the response structure
        data = response.json()
        assert "id" in data
        assert "content" in data
        assert "citations" in data
        assert "response_type" in data
        assert "timestamp" in data


def test_chat_endpoint_missing_query():
    """
    Test the contract for the /chat endpoint with missing query.
    """
    response = client.post(
        "/chat",
        json={},
        headers={"X-API-Key": "test-key"}
    )
    
    # Should return a validation error
    assert response.status_code == 422


def test_text_analysis_endpoint_contract():
    """
    Test the contract for the /chat/text-analysis endpoint.
    """
    # Mock the services to test only the API contract
    with patch('src.services.rag_service.RAGService') as mock_rag_service:
        # Mock the response
        mock_response = Mock()
        mock_response.id = "test_response_id"
        mock_response.content = "This is a test response"
        mock_response.citations = []
        mock_response.response_type = "using_selected_text"
        mock_response.timestamp = "2023-10-20T10:00:00Z"
        
        mock_rag_service_instance = Mock()
        mock_rag_service_instance.process_selected_text.return_value = mock_response
        mock_rag_service.return_value = mock_rag_service_instance
        
        # Make a request to the endpoint
        response = client.post(
            "/chat/text-analysis",
            json={
                "text": "Robot kinematics is the study of motion in robotic systems..."
            },
            headers={"X-API-Key": "test-key"}
        )
        
        # Verify the response status
        assert response.status_code == 200
        
        # Verify the response structure
        data = response.json()
        assert "id" in data
        assert "content" in data
        assert "citations" in data
        assert "response_type" in data
        assert "timestamp" in data


def test_health_endpoint_contract():
    """
    Test the contract for the /health endpoint.
    """
    response = client.get("/health")

    # Verify the response status
    assert response.status_code == 200

    # Verify the response structure
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert data["status"] in ["healthy", "degraded", "unavailable"]


def test_graceful_degradation_responses():
    """
    Test the contract for graceful degradation responses when services fail.
    """
    # This test verifies that when there are failures in the RAG pipeline,
    # the API still returns responses with appropriate fallback messages
    response = client.post(
        "/chat",
        json={"query": "What are the fundamentals of robot kinematics?"},
        headers={"X-API-Key": "test-key"}
    )

    # The response should still be successful (200) even if internal services have issues
    assert response.status_code == 200

    data = response.json()

    # Verify the response structure is maintained
    assert "id" in data
    assert "content" in data
    assert "citations" in data
    assert "response_type" in data
    assert "timestamp" in data

    # Verify content has appropriate fallback if needed
    # The response_type should indicate the type of response provided
    assert data["response_type"] in ["from_book", "fallback_general", "fallback_unavailable", "using_selected_text"]

    # If it's a fallback response, ensure it has the appropriate message format
    if data["response_type"] in ["fallback_general", "fallback_unavailable"]:
        content = data["content"].lower()
        if "fallback_unavailable" in data["response_type"]:
            assert "unable to access book content" in content or "retrieval service unavailable" in content
        elif "fallback_general" in data["response_type"]:
            assert "no direct book content found" in content


@patch('src.services.rag_service.RAGService')
def test_contract_response_with_qdrant_failure(mock_rag_service):
    """
    Test the contract when Qdrant service fails but API still responds gracefully.
    """
    from unittest.mock import Mock

    # Mock the RAG service to simulate Qdrant failure
    mock_response = Mock()
    mock_response.id = "test_response_id"
    mock_response.content = "Unable to access book content at this moment (retrieval service unavailable). Here's a helpful answer based on general knowledge: General information about robotics."
    mock_response.citations = []
    mock_response.response_type = "fallback_unavailable"
    mock_response.timestamp = "2023-10-20T10:00:00Z"

    mock_rag_service_instance = Mock()
    mock_rag_service_instance.process_query.return_value = mock_response
    mock_rag_service.return_value = mock_rag_service_instance

    # Make a request to the endpoint
    response = client.post(
        "/chat",
        json={"query": "What are the fundamentals of robot kinematics?"},
        headers={"X-API-Key": "test-key"}
    )

    # Verify the response status is still successful
    assert response.status_code == 200

    # Verify the response structure
    data = response.json()
    assert "id" in data
    assert "content" in data
    assert data["response_type"] == "fallback_unavailable"
    assert len(data["citations"]) == 0  # No citations in fallback
    assert "Unable to access book content" in data["content"]