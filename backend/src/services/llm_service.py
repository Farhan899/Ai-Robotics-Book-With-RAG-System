"""
LLM Service for the RAG Chatbot for Robotics Book.
Handles interactions with the OpenRouter API for language model calls.
"""
from typing import List, Optional
from openai import AsyncOpenAI
from src.core.config import settings
from src.core.logging import logger


class LLMService:
    """
    Service class for interacting with the OpenRouter API to generate responses.
    """
    
    def __init__(self):
        """
        Initialize the LLM Service with OpenRouter API configuration.
        """
        self.client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
    
    async def generate_response(
        self, 
        query: str, 
        contexts: List[str] = None, 
        selected_text: str = None
    ) -> str:
        """
        Generate a response using the LLM based on the query and provided context.
        
        Args:
            query: The user's question or query text
            contexts: Optional list of context strings retrieved from the vector database
            selected_text: Optional user-provided text to analyze instead of using contexts
            
        Returns:
            Generated response string
        """
        try:
            # Build the prompt based on the input type
            if selected_text:
                # Use selected text as context
                prompt = self._build_selected_text_prompt(query, selected_text)
            elif contexts and len(contexts) > 0:
                # Use retrieved contexts
                prompt = self._build_contextual_prompt(query, contexts)
            else:
                # No context provided - general response
                prompt = self._build_general_prompt(query)
            
            # Create the messages for the API call
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Make the API call to OpenRouter
            response = await self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": settings.site_url,
                    "X-Title": settings.site_name,
                },
                model=settings.openrouter_model,
                messages=messages
            )
            
            # Extract and return the generated content
            content = response.choices[0].message.content
            
            # Log the interaction
            logger.info(f"LLM API call successful, response length: {len(content) if content else 0} characters")
            
            return content
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {str(e)}", exc_info=True)
            raise e
    
    def _build_contextual_prompt(self, query: str, contexts: List[str]) -> str:
        """
        Build a prompt that includes retrieved contexts.
        
        Args:
            query: The user's question
            contexts: List of context strings retrieved from the database
            
        Returns:
            Formatted prompt string
        """
        context_str = "\n\n".join(contexts)
        return (
            f"You are an AI assistant for a robotics book. Use the following retrieved context to answer the user's question.\n\n"
            f"Retrieved context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Provide a helpful, accurate answer based on the context. If the context doesn't contain the answer, say so and provide a general answer. "
            f"Cite the source with the format 'Book Section: [relevant metadata if available]' where possible."
        )
    
    def _build_selected_text_prompt(self, query: str, selected_text: str) -> str:
        """
        Build a prompt that uses user-selected text as context.
        
        Args:
            query: The user's question about the selected text
            selected_text: The text provided by the user for analysis
            
        Returns:
            Formatted prompt string
        """
        return (
            f"You are an AI assistant for a robotics book. Analyze the following text provided by the user and answer their question.\n\n"
            f"Selected text: {selected_text}\n\n"
            f"Question: {query}\n\n"
            f"Provide an analysis based solely on the selected text. Do not use any other knowledge sources."
        )
    
    def _build_general_prompt(self, query: str) -> str:
        """
        Build a general prompt for when no context is available.
        
        Args:
            query: The user's question
            
        Returns:
            Formatted prompt string
        """
        return (
            f"You are an AI assistant for a robotics book. The user asked: '{query}'\n\n"
            f"Provide a helpful answer based on your knowledge of robotics. "
            f"If possible, reference principles that would typically be covered in a robotics book. "
            f"Note that this response is based on general knowledge, not specific book content."
        )