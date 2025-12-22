"""
Chat router for the RAG Chatbot for Robotics Book.
Defines the API endpoints for chat functionality.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional
import uuid
from src.models.query import QueryRequest
from src.models.response import ChatResponse, ErrorResponse
from src.services.rag_service import RAGService
from src.services.qdrant_service import QdrantService
from src.services.llm_service import LLMService
from src.core.config import settings
from src.core.logging import logger


# Create the router
chat_router = APIRouter()


def get_rag_service() -> RAGService:
    """
    Dependency function to get the RAG service instance.
    """
    # Initialize services
    qdrant_service = QdrantService()
    llm_service = LLMService()
    
    # Create and return the RAG service
    return RAGService(
        qdrant_service=qdrant_service,
        llm_service=llm_service
    )


@chat_router.post("/", response_model=ChatResponse)
async def chat(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> ChatResponse:
    """
    Endpoint to submit a query to the RAG chatbot.
    
    Args:
        request: The query request containing the user's question
        rag_service: The RAG service to process the query
        
    Returns:
        ChatResponse: The response to the user's query
    """
    try:
        # Log the incoming request
        req_id = str(uuid.uuid4())
        logger.info(f"Processing chat request {req_id}: {request.query[:50]}...")
        
        # Process the query through the RAG service
        response = await rag_service.process_query(request)
        
        # Log the successful response
        logger.info(f"Chat request {req_id} processed successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "CHAT_PROCESSING_ERROR",
                "message": "Error processing the chat query"
            }
        )


@chat_router.post("/text-analysis", response_model=ChatResponse)
async def analyze_selected_text(
    request: Request,
    rag_service: RAGService = Depends(get_rag_service)
) -> ChatResponse:
    """
    Endpoint to analyze user-provided text without querying the database.
    
    Args:
        request: The request containing the text to analyze and optional query
        rag_service: The RAG service to process the text analysis
        
    Returns:
        ChatResponse: The analysis of the provided text
    """
    try:
        # Parse the request body
        body = await request.json()
        text = body.get("text")
        query = body.get("query", "Please explain this text")
        
        # Validate the input
        if not text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "MISSING_TEXT",
                    "message": "Text to analyze is required"
                }
            )
        
        # Log the incoming request
        req_id = str(uuid.uuid4())
        logger.info(f"Processing text analysis request {req_id}: {text[:50]}...")
        
        # Process the selected text through the RAG service
        response = await rag_service.process_selected_text(text, query)
        
        # Log the successful response
        logger.info(f"Text analysis request {req_id} processed successfully")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing text analysis request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "TEXT_ANALYSIS_ERROR",
                "message": "Error processing the text analysis"
            }
        )


# Add validation and error handling for user story 1 (T022 would be addressed in the implementation above)

# Add logging for user story 1 operations (T023 is addressed by using the logger throughout)

# Response citation logic is implemented in rag_service.py (T024)