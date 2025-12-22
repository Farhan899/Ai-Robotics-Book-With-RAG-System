"""
Script to initialize Qdrant collection and populate it with robotics book content.
"""
import asyncio
import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from src.core.config import settings


async def initialize_qdrant():
    """Initialize Qdrant collection and populate with sample data."""
    
    # Initialize the Qdrant client (use local if no URL provided)
    if settings.qdrant_url:
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    else:
        client = AsyncQdrantClient(
            path=settings.qdrant_local_path
        )
    
    # Create the collection if it doesn't exist
    try:
        # Check if collection exists
        collections = await client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if settings.qdrant_collection_name not in collection_names:
            print(f"Creating collection: {settings.qdrant_collection_name}")

            # Create a collection with vector configuration
            # Using 384 dimensions to match the embedding model (BAAI/bge-small-en-v1.5)
            await client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Using 384 dimensions to match our embedding model
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection {settings.qdrant_collection_name} created successfully")
        else:
            print(f"Collection {settings.qdrant_collection_name} already exists, deleting and recreating...")
            await client.delete_collection(settings.qdrant_collection_name)
            await client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Using 384 dimensions to match our embedding model
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection {settings.qdrant_collection_name} recreated successfully")
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        return False
    
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
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Robotic manipulation involves controlling robot arms to interact with objects. This includes inverse kinematics to calculate joint angles needed to reach a position, grasp planning to determine how to grip objects, and trajectory planning to move the arm smoothly and avoid obstacles.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 6",
                "section": "6.1 Manipulation Fundamentals",
                "topic": "Manipulation"
            }
        },
        {
            "id": str(uuid.uuid4()),
            "content": "A digital twin is a virtual representation of a physical system or process. In robotics, digital twins allow you to simulate, analyze, and optimize robot behavior before implementing it in the real world. This helps reduce development time, test scenarios safely, and improve overall system performance.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 4",
                "section": "4.3 Digital Twins in Robotics",
                "topic": "Digital Twins"
            }
        },
        {
            "id": str(uuid.uuid4()),
            "content": "Vision-Language-Action (VLA) models connect visual perception with language understanding and physical actions. These models allow robots to interpret natural language commands, understand their visual environment, and execute appropriate actions. Examples include models that can follow instructions like 'Pick up the red block' by perceiving the scene, understanding the command, and executing the action.",
            "metadata": {
                "source": "Introduction to Robotics",
                "chapter": "Chapter 9",
                "section": "9.2 Vision-Language-Action Models",
                "topic": "VLA Models"
            }
        }
    ]
    
    # Prepare points to upload
    points = []
    for doc in sample_documents:
        # For simplicity, we're not using embeddings here.
        # In a real implementation, you would generate embeddings for the content.
        # Here we'll use a placeholder vector of correct size - in reality, use proper embeddings
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=[0.0] * 384,  # Placeholder vector - in reality, use proper embeddings
                payload={
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                }
            )
        )
    
    # Upload points to the collection
    try:
        await client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points
        )
        print(f"Successfully uploaded {len(points)} documents to Qdrant collection: {settings.qdrant_collection_name}")
        return True
    except Exception as e:
        print(f"Error uploading documents: {str(e)}")
        return False


if __name__ == "__main__":
    asyncio.run(initialize_qdrant())