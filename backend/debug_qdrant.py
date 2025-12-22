"""
Debug script to test Qdrant connection and search.
"""
import asyncio
from src.services.qdrant_service import QdrantService
from src.core.config import settings


async def debug_qdrant():
    """Debug Qdrant connection and search functionality."""
    
    print(f"Using Qdrant path: {settings.qdrant_local_path}")
    print(f"Using collection: {settings.qdrant_collection_name}")
    
    # Initialize the Qdrant service
    qdrant_service = QdrantService()
    
    print("Testing connection...")
    try:
        is_connected = await qdrant_service.check_connection()
        print(f"Connection status: {is_connected}")
    except Exception as e:
        print(f"Connection error: {str(e)}")
        return
    
    print("\nTesting search...")
    try:
        results = await qdrant_service.search_similar("What is ROS?", limit=5)
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  ID: {result.id}")
            print(f"  Content: {result.content[:100]}...")
            print(f"  Similarity: {result.similarity_score}")
            print(f"  Metadata: {result.metadata}")
            print()
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_qdrant())