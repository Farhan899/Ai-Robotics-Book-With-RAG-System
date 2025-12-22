# Quickstart Guide: RAG Chatbot for Robotics Book

## Overview
This guide provides instructions for setting up and running the RAG Chatbot for Robotics Book project. This system allows users to ask questions about robotics content from a book and receive AI-generated responses based on the book's content.

## Prerequisites
- Python 3.11 or higher
- pip package manager
- Git version control
- Access to Qdrant vector database (Cloud Free Tier)
- OpenRouter API key for z-ai/glm-4.5-air:free model

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn python-dotenv openai qdrant-client PyYAML pytest
```

### 4. Configure Environment Variables
Create a `.env` file in the project root with the following:
```env
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=z-ai/glm-4.5-air:free
YOUR_SITE_URL=https://yourdomain.com
YOUR_SITE_NAME=YourSiteName
```

### 5. Set Up Qdrant Collection
Ensure you have the `robotics_docs` collection set up in your Qdrant instance with appropriate vector dimensions for your embeddings.

## Running the Application

### 1. Start the Development Server
```bash
cd backend
uvicorn src.api.main:app --reload --port 8000
```

### 2. Verify the API is Running
Visit `http://localhost:8000/health` to check if the service is running properly.

### 3. Test the Chat Endpoint
You can test the API using curl:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "What are the fundamentals of robotics?"
  }'
```

## API Usage Examples

### Standard Query
Request:
```json
{
  "query": "Explain robot kinematics"
}
```

Response:
```json
{
  "id": "resp_1234567890",
  "content": "Robot kinematics is the study of motion in robotic systems, focusing on the relationship between joint parameters and the position and orientation of the robot's end-effector...",
  "citations": ["Book Section: Chapter 3 - Kinematics"],
  "response_type": "from_book",
  "timestamp": "2023-10-20T10:00:00Z"
}
```

### Query with Selected Text
Request:
```json
{
  "query": "Explain this concept",
  "selected_text": "Robot kinematics is the study of motion in robotic systems..."
}
```

Response:
```json
{
  "id": "resp_1234567891",
  "content": "This text explains how robot kinematics deals with the mathematical relationships that describe the position and movement of robotic systems...",
  "citations": [],
  "response_type": "using_selected_text",
  "timestamp": "2023-10-20T10:01:00Z"
}
```

## Testing

### Run Unit Tests
```bash
python -m pytest tests/unit/ -v
```

### Run Integration Tests
```bash
python -m pytest tests/integration/ -v
```

### Run All Tests
```bash
python -m pytest tests/ -v
```

## Architecture Overview

The application follows a service-layer architecture with clear separation of concerns:

- `src/models/` - Data models and validation
- `src/services/` - Business logic implementation
- `src/api/` - API endpoints and routing
- `src/core/` - Configuration and core utilities
- `tests/` - Unit, integration, and contract tests

## Troubleshooting

### Qdrant Connection Issues
- Verify your Qdrant URL and API key are correct
- Check that the `robotics_docs` collection exists
- Ensure your network allows connections to Qdrant

### API Limitations
- The system will gracefully handle Qdrant outages by switching to general knowledge responses
- If no relevant content is found in the database, the system will provide a general helpful response

### Performance Issues
- Monitor response times; the target is 95% of queries responding within 5 seconds
- Consider implementing caching for frequently asked questions