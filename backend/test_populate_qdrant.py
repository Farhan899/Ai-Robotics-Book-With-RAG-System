"""
Test script to populate Qdrant with sample robotics book content.
This will allow the RAG system to retrieve and use actual book content.
"""
import asyncio
import uuid
from src.services.qdrant_service import QdrantService
from src.core.config import settings


async def populate_qdrant():
    """Populate Qdrant with sample robotics book content."""
    
    # Initialize the Qdrant service
    qdrant_service = QdrantService()
    
    # Test the connection first
    try:
        is_connected = await qdrant_service.check_connection()
        if not is_connected:
            print("Error: Could not connect to Qdrant. Please check your configuration.")
            print(f"URL: {settings.qdrant_url}")
            print(f"Collection: {settings.qdrant_collection_name}")
            return
        else:
            print("Successfully connected to Qdrant")
    except Exception as e:
        print(f"Error connecting to Qdrant: {str(e)}")
        return
    
    # Sample robotics book content
    sample_documents = [
        {
            "id": str(uuid.uuid4()),
            "content": "ROS (Robot Operating System) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. ROS 2 uses DDS (Data Distribution Service) for communication between nodes.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 1",
                "section": "1.1 What is ROS",
                "topic": "ROS Fundamentals"
            }
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Gazebo is a robot simulation environment that provides physics simulation, sensor simulation, and realistic environments for testing robotic systems before deployment. It integrates well with ROS and allows you to test your robot algorithms in a safe, virtual environment before running them on real hardware.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 3",
                "section": "3.2 Simulation with Gazebo",
                "topic": "Simulation"
            }
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Robot navigation typically involves several components: global path planning (finding the optimal path from start to goal), local path planning (avoiding obstacles while following the global path), localization (determining the robot's position), and mapping (creating a representation of the environment). In ROS, the Navigation2 stack provides these capabilities.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 8",
                "section": "8.1 Navigation Systems",
                "topic": "Navigation"
            }
        },
        {
            "id": str(uuid.uuid4()),
            "content": "SLAM (Simultaneous Localization and Mapping) is a technique that allows robots to build a map of an unknown environment while simultaneously keeping track of their location within that environment. Common approaches include EKF SLAM, FastSLAM, and graph-based SLAM. In ROS, packages like Cartographer and RTAB-Map provide SLAM capabilities.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 7",
                "section": "7.3 SLAM Techniques",
                "topic": "SLAM"
            }
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Computer vision in robotics involves processing camera images to understand the environment. This can include object detection, tracking, SLAM (Simultaneous Localization and Mapping), and depth estimation. Libraries like OpenCV and deep learning frameworks like PyTorch/TensorFlow are commonly used for vision tasks in robotics.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 5",
                "section": "5.1 Computer Vision in Robotics",
                "topic": "Computer Vision"
            }
        }
    ]
    
    # Add documents to Qdrant
    for doc in sample_documents:
        success = await qdrant_service.add_document(
            document_id=doc["id"],
            content=doc["content"],
            metadata=doc["metadata"]
        )
        
        if success:
            print(f"Successfully added document: {doc['metadata']['section']}")
        else:
            print(f"Failed to add document: {doc['metadata']['section']}")
    
    print("\nQdrant population completed!")


if __name__ == "__main__":
    asyncio.run(populate_qdrant())