"""
Unit tests for the LLM Service of the RAG Chatbot for Robotics Book.
"""
import pytest
import openai
from unittest.mock import Mock, AsyncMock, patch
from src.services.llm_service import LLMService
from src.core.config import settings


@pytest.fixture
def llm_service():
    """Create an LLMService instance."""
    return LLMService()


@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
async def test_generate_response_from_context(mock_openai_class, llm_service):
    """Test generating a response using retrieved context."""
    # Arrange
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Based on the book, robot kinematics is the study of motion in robotic systems..."
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    mock_openai_class.return_value = mock_client

    retrieved_contexts = [
        "Robot kinematics is the study of motion in robotic systems...",
        "It focuses on the mathematical relationships that describe the position and movement of robotic systems."
    ]
    query = "What are the fundamentals of robot kinematics?"

    # Act
    result = await llm_service.generate_response(query, retrieved_contexts)

    # Assert
    assert result == "Based on the book, robot kinematics is the study of motion in robotic systems..."
    mock_openai_class.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()
    
    # Check that the call includes the correct parameters
    call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]['model'] == settings.openrouter_model
    assert call_args[1]['messages'][0]['content'] is not None
    # Verify the prompt contains both context and query
    prompt_content = call_args[1]['messages'][0]['content']
    assert "retrieved context" in prompt_content.lower()
    assert query in prompt_content


@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
async def test_generate_response_without_context(mock_openai_class, llm_service):
    """Test generating a response without any retrieved context (fallback)."""
    # Arrange
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Robot kinematics is the study of motion in robotic systems."
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    mock_openai_class.return_value = mock_client

    query = "What are the fundamentals of robot kinematics?"

    # Act
    result = await llm_service.generate_response(query)

    # Assert
    assert result == "Robot kinematics is the study of motion in robotic systems."
    mock_openai_class.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
async def test_generate_response_with_selected_text(mock_openai_class, llm_service):
    """Test generating a response based on user-selected text."""
    # Arrange
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Based on the selected text, robot kinematics deals with motion in robotic systems..."
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    mock_openai_class.return_value = mock_client

    selected_text = "Robot kinematics is the study of motion in robotic systems..."
    query = "Explain this concept"

    # Act
    result = await llm_service.generate_response(query, selected_text=selected_text)

    # Assert
    assert result == "Based on the selected text, robot kinematics deals with motion in robotic systems..."
    mock_openai_class.assert_called_once()
    mock_client.chat.completions.create.assert_called_once()
    
    # Check that the call includes the selected text in the prompt
    call_args = mock_client.chat.completions.create.call_args
    prompt_content = call_args[1]['messages'][0]['content']
    assert selected_text in prompt_content
    assert query in prompt_content


@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
async def test_generate_response_api_error(mock_openai_class, llm_service):
    """Test handling of API errors during response generation."""
    # Arrange
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(side_effect=openai.APIError("API Error"))
    mock_openai_class.return_value = mock_client

    query = "What are the fundamentals of robot kinematics?"

    # Act & Assert
    with pytest.raises(openai.APIError):
        await llm_service.generate_response(query)


@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
async def test_generate_response_timeout(mock_openai_class, llm_service):
    """Test handling of timeouts during response generation."""
    # Arrange
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(side_effect=TimeoutError("Request timed out"))
    mock_openai_class.return_value = mock_client

    query = "What are the fundamentals of robot kinematics?"

    # Act & Assert
    with pytest.raises(TimeoutError):
        await llm_service.generate_response(query)