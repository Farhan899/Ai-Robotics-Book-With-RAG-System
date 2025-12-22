"""
Configuration management for the RAG Chatbot for Robotics Book.
"""
from pydantic_settings import SettingsConfigDict, BaseSettings


class Settings(BaseSettings):
    """
    Application settings using Pydantic Settings.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    # Qdrant settings
    qdrant_api_key: str = ""
    qdrant_url: str = ""
    qdrant_collection_name: str = "robotics_docs"
    qdrant_local_path: str = "./qdrant_data"  # Path for local Qdrant instance
    
    # OpenRouter settings
    openrouter_api_key: str
    openrouter_model: str = "z-ai/glm-4.5-air:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Application settings
    app_title: str = "RAG Chatbot for Robotics Book"
    app_description: str = "API for a Retrieval-Augmented Generation chatbot that answers questions about robotics content from a book"
    app_version: str = "1.0.0"
    
    # Performance settings
    response_timeout_seconds: int = 5
    
    # Site settings for OpenRouter API
    site_url: str = "https://yourdomain.com"
    site_name: str = "RAG-Chatbot"


# Create a single instance of settings
settings = Settings()