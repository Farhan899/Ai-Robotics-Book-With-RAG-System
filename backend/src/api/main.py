"""
Main FastAPI application entry point for the RAG Chatbot for Robotics Book.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .chat_router import chat_router
from .health_router import health_router

app = FastAPI(
    title="RAG Chatbot for Robotics Book",
    description="API for a Retrieval-Augmented Generation chatbot that answers questions about robotics content from a book",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose additional headers that might be needed
    expose_headers=["X-Response-Time"]
)

# Include routers
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(health_router, tags=["health"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)