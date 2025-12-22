"""
Main FastAPI application entry point for the RAG Chatbot for Robotics Book.
"""
from fastapi import FastAPI
from .chat_router import chat_router
from .health_router import health_router

app = FastAPI(
    title="RAG Chatbot for Robotics Book",
    description="API for a Retrieval-Augmented Generation chatbot that answers questions about robotics content from a book",
    version="1.0.0"
)

# Include routers
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(health_router, tags=["health"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)