# RAG Chatbot for Robotics Book

This project implements a Retrieval-Augmented Generation (RAG) chatbot for a robotics book. The system queries a Qdrant vector database for semantically similar content before generating responses using the OpenRouter API.

## Features

- Ask questions about robotics concepts from a book
- Get answers based on book content with proper citations
- Analyze user-provided text from the book
- Graceful fallback when the RAG system is unavailable
- Proper error handling and logging

## Architecture

The system follows a service-oriented architecture with clear separation of concerns:

- `src/models/` - Data models and validation
- `src/services/` - Business logic implementation
- `src/api/` - API endpoints and routing
- `src/core/` - Configuration and core utilities
- `tests/` - Unit, integration, and contract tests

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with the following:
   ```
   QDRANT_API_KEY=your_qdrant_api_key
   QDRANT_URL=your_qdrant_url
   OPENROUTER_API_KEY=your_openrouter_api_key
   SITE_URL=https://yourdomain.com
   SITE_NAME=YourSiteName
   ```
3. Run the application: `python -m src.api.main` or `uvicorn src.api.main:app --reload`

## API Endpoints

- `POST /chat` - Submit a query to the RAG chatbot
- `POST /chat/text-analysis` - Analyze user-provided text
- `GET /health` - Health check endpoint

## Testing

Run all tests with: `pytest`

Run specific test types:
- Unit tests: `pytest tests/unit/`
- Integration tests: `pytest tests/integration/`
- Contract tests: `pytest tests/contract/`

## Configuration

The application uses Pydantic Settings for configuration management. Settings can be loaded from environment variables or a .env file.