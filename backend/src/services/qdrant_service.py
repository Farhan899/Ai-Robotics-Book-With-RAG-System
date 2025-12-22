"""
Qdrant Service for the RAG Chatbot for Robotics Book.
Handles interactions with the Qdrant vector database.
"""
from typing import List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.models.document import RetrievedContext
from src.core.config import settings
from src.core.logging import logger
from fastembed import TextEmbedding


class QdrantService:
    """
    Service class for interacting with Qdrant vector database.
    """
    
    def __init__(self, client: AsyncQdrantClient = None, collection_name: str = None):
        """
        Initialize the Qdrant Service.

        Args:
            client: AsyncQdrantClient instance (if not provided, creates a new one)
            collection_name: Name of the Qdrant collection to use (defaults to settings)
        """
        if client is None:
            # Use local Qdrant instance if no URL is provided
            if settings.qdrant_url:
                # Use remote Qdrant instance
                self.client = AsyncQdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                )
            else:
                # Use local Qdrant instance
                self.client = AsyncQdrantClient(
                    path=settings.qdrant_local_path
                )
        else:
            self.client = client

        self.collection_name = collection_name or settings.qdrant_collection_name

        # Initialize the text embedding model
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    async def search_similar(self, query_text: str, limit: int = 5) -> List[RetrievedContext]:
        """
        Search for similar content in the Qdrant database.

        Args:
            query_text: The text to search for similar content to
            limit: Maximum number of results to return

        Returns:
            List of RetrievedContext objects containing the similar content
        """
        try:
            # Generate embedding for the query text
            query_embedding = None
            for embedding in self.embedding_model.embed([query_text]):
                query_embedding = embedding.tolist()
                break  # Get the first embedding

            if query_embedding is None:
                raise Exception("Failed to generate embedding for query text")

            # Perform semantic search in the Qdrant collection
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert the search results to RetrievedContext objects
            retrieved_contexts = []
            for result in results:
                # Extract content and metadata from the payload
                payload = result.payload or {}
                content = payload.get("content", "")
                metadata = payload.get("metadata", {})
                
                # Create a RetrievedContext object
                context = RetrievedContext(
                    id=str(result.id),
                    content=content,
                    metadata=metadata,
                    similarity_score=result.score,
                    source_document_id=str(result.id)  # Using the Qdrant ID as source document ID
                )
                
                retrieved_contexts.append(context)
            
            return retrieved_contexts
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}", exc_info=True)
            raise e
    
    async def add_document(self, document_id: str, content: str, metadata: dict, vector: List[float] = None) -> bool:
        """
        Add a document to the Qdrant collection.
        
        Args:
            document_id: Unique identifier for the document
            content: The text content of the document
            metadata: Additional metadata about the document
            vector: The embedding vector for the document (if not provided, Qdrant will generate it)
            
        Returns:
            True if the document was added successfully, False otherwise
        """
        try:
            # Prepare the payload
            payload = {
                "content": content,
                "metadata": metadata
            }
            
            # Prepare the points to be upserted
            points = [
                models.PointStruct(
                    id=document_id,
                    vector=vector or [],  # If no vector is provided, it's assumed Qdrant will handle it via inference API
                    payload=payload
                )
            ]
            
            # Upsert the points into the collection
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Document {document_id} added to collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {document_id} to Qdrant: {str(e)}", exc_info=True)
            return False
    
    async def check_connection(self) -> bool:
        """
        Check if we can connect to the Qdrant database.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Try to get collection info to test the connection
            await self.client.get_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {str(e)}", exc_info=True)
            return False