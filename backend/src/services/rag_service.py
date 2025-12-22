"""
RAG Service for the RAG Chatbot for Robotics Book.
Orchestrates retrieval and generation processes.
"""
import uuid
from datetime import datetime
from typing import List, Optional
from src.services.qdrant_service import QdrantService
from src.services.llm_service import LLMService
from src.models.query import QueryRequest
from src.models.response import ChatResponse, ResponseType
from src.models.document import RetrievedContext
from src.core.logging import logger


class RAGService:
    """
    Service class that orchestrates the Retrieval-Augmented Generation process.
    """
    
    def __init__(self, qdrant_service: QdrantService, llm_service: LLMService):
        """
        Initialize the RAG Service with required dependencies.
        
        Args:
            qdrant_service: Service for interacting with Qdrant vector database
            llm_service: Service for interacting with LLM API
        """
        self.qdrant_service = qdrant_service
        self.llm_service = llm_service
    
    async def process_query(self, query_request: QueryRequest) -> ChatResponse:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query_request: The query request containing the user's question
            
        Returns:
            ChatResponse: The response to the user's query
        """
        query_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        try:
            # If selected text is provided, use it directly without Qdrant retrieval
            if query_request.selected_text:
                logger.info(f"Processing query {query_id} with user-selected text")
                
                # Generate response based on the selected text
                content = await self.llm_service.generate_response(
                    query_request.query or "Please explain this text", 
                    selected_text=query_request.selected_text
                )
                
                return ChatResponse(
                    id=query_id,
                    content=content,
                    citations=[],
                    response_type=ResponseType.using_selected_text,
                    timestamp=timestamp
                )
            
            # Otherwise, search for relevant content in Qdrant
            logger.info(f"Searching for relevant content for query {query_id}")
            retrieved_contexts = await self.qdrant_service.search_similar(
                query_request.query, 
                limit=5
            )
            
            # If we found relevant content, use it to generate the response
            if retrieved_contexts:
                logger.info(f"Found {len(retrieved_contexts)} relevant contexts for query {query_id}")
                
                # Prepare context content for the LLM
                context_contents = [ctx.content for ctx in retrieved_contexts]
                
                # Generate response using the retrieved context
                content = await self.llm_service.generate_response(
                    query_request.query, 
                    context_contents
                )
                
                # Create citations from the metadata
                citations = [
                    f"Book Section: {ctx.metadata.get('source', 'Unknown')} - {ctx.metadata.get('chapter', '')} {ctx.metadata.get('section', '')}".strip()
                    for ctx in retrieved_contexts
                ]
                
                # Filter out empty citations
                citations = [citation for citation in citations if citation != "Book Section:"]
                
                return ChatResponse(
                    id=query_id,
                    content=content,
                    citations=citations,
                    response_type=ResponseType.from_book,
                    timestamp=timestamp
                )
            else:
                logger.info(f"No relevant content found for query {query_id}, using general knowledge")
                
                # No relevant content found, generate a general response
                content = await self.llm_service.generate_response(query_request.query)
                content = f"No direct book content found for this query. Here's a helpful overview: {content}"
                
                return ChatResponse(
                    id=query_id,
                    content=content,
                    citations=[],
                    response_type=ResponseType.fallback_general,
                    timestamp=timestamp
                )
                
        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}", exc_info=True)
            
            # Return a fallback response if an error occurs
            content = f"Unable to access book content at this moment (retrieval service unavailable). Here's a helpful answer based on general knowledge: {await self.llm_service.generate_response(query_request.query)}"
            
            return ChatResponse(
                id=query_id,
                content=content,
                citations=[],
                response_type=ResponseType.fallback_unavailable,
                timestamp=timestamp
            )
    
    async def process_selected_text(self, text: str, query: Optional[str] = None) -> ChatResponse:
        """
        Process user-provided text for analysis without querying the database.
        
        Args:
            text: The text provided by the user for analysis
            query: Optional specific question about the provided text
            
        Returns:
            ChatResponse: The response based on the selected text
        """
        query_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        try:
            # Generate response based on the selected text
            final_query = query or "Please explain this text"
            content = await self.llm_service.generate_response(final_query, selected_text=text)
            
            return ChatResponse(
                id=query_id,
                content=content,
                citations=[],
                response_type=ResponseType.using_selected_text,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Error processing selected text {query_id}: {str(e)}", exc_info=True)
            content = f"Unable to analyze the provided text at this moment. Here's a general response: {await self.llm_service.generate_response(query or 'Please explain this text')}"
            
            return ChatResponse(
                id=query_id,
                content=content,
                citations=[],
                response_type=ResponseType.fallback_unavailable,
                timestamp=timestamp
            )