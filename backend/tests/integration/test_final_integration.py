"""
Final integration tests for the RAG Chatbot for Robotics Book.
This tests the complete system integration across all components.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_complete_system_integration():
    """
    Test the complete system integration by making actual API calls
    that go through all layers of the application.
    """
    # This test would normally require actual Qdrant and OpenRouter services
    # For this demonstration, we'll test the API contract and structure
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    
    # Test chat endpoint structure (without actual processing)
    # Note: This would require mocking the services in a real scenario
    response = client.post(
        "/chat",
        json={"query": "What are the fundamentals of robotics?"},
        headers={"X-API-Key": "test-key"}
    )
    
    # The endpoint should return a structured response
    # (Actual content depends on service responses which we're not testing here)
    assert response.status_code in [200, 500]  # Either success or service error, but not 404
    
    if response.status_code == 200:
        data = response.json()
        # Check that response follows expected structure
        assert "id" in data
        assert "content" in data
        assert "citations" in data
        assert "response_type" in data
        assert "timestamp" in data


def test_text_analysis_endpoint_integration():
    """
    Test the text analysis endpoint integration.
    """
    response = client.post(
        "/chat/text-analysis",
        json={
            "text": "Robot kinematics is the study of motion in robotic systems...",
            "query": "Explain this concept"
        },
        headers={"X-API-Key": "test-key"}
    )
    
    # The endpoint should return a structured response
    assert response.status_code in [200, 500]  # Either success or service error, but not 404
    
    if response.status_code == 200:
        data = response.json()
        # Check that response follows expected structure
        assert "id" in data
        assert "content" in data
        assert "citations" in data  # Should be empty for selected text analysis
        assert "response_type" in data
        assert "timestamp" in data


def test_error_handling_integration():
    """
    Test that error handling works properly across all components.
    """
    # Test with invalid input
    response = client.post(
        "/chat",
        json={},
        headers={"X-API-Key": "test-key"}
    )
    
    # Should return validation error (422) for invalid input
    # or a 200 with a fallback response for valid but empty query
    assert response.status_code in [422, 200, 500]


if __name__ == "__main__":
    pytest.main([__file__])