# ROS 2 Architecture and DDS Middleware

## Introduction to ROS 2 Architecture

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software that provides services designed for a heterogeneous computer cluster. The architecture of ROS 2 is fundamentally different from ROS 1, primarily due to its use of the Data Distribution Service (DDS) middleware.

## DDS Middleware

### What is DDS?

Data Distribution Service (DDS) is a middleware protocol and API standard for distributed, real-time, dependable data exchanges. DDS provides a publish-subscribe pattern that allows for efficient communication between different components of a robotic system.

### DDS Quality of Service (QoS) Policies

DDS provides Quality of Service (QoS) policies that allow fine-grained control over communication behavior:

- **Reliability**: Ensures delivery of data
  - `RELIABLE`: All messages are delivered
  - `BEST_EFFORT`: Messages may be lost but delivery is not guaranteed

- **Durability**: Determines if late-joining subscribers receive historical data
  - `TRANSIENT_LOCAL`: Late-joining subscribers receive historical data
  - `VOLATILE`: Only future data is sent to late-joining subscribers

- **History**: Controls how many samples are kept for each topic
  - `KEEP_LAST`: Keeps the last n samples
  - `KEEP_ALL`: Keeps all samples (subject to resource limits)

- **Depth**: Number of samples to keep in history (used with `KEEP_LAST`)

## ROS 2 Core Concepts

### Nodes

Nodes are the fundamental building blocks of a ROS 2 program. Each node is a process that performs computation. In ROS 2, nodes are designed to be modular and can be distributed across multiple machines.

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
```

### Topics and Messages

Topics are named buses over which nodes exchange messages. Messages are the data structures that are passed between nodes. ROS 2 uses a strongly-typed message system with predefined message types.

### Services

Services provide synchronous request/response communication between nodes. A service has a request and response message type.

```python
from example_interfaces.srv import AddTwoInts

def add_two_ints_callback(request, response):
    response.sum = request.a + request.b
    return response
```

### Actions

Actions provide asynchronous request/goal-based communication with feedback and status updates. They are ideal for long-running tasks where you want to monitor progress.

## Client Library Architecture

### rclpy and rclcpp

ROS 2 provides client libraries for different programming languages:
- `rclpy`: Python client library
- `rclcpp`: C++ client library

These libraries provide a common interface to the underlying DDS implementation while allowing language-specific features.

### ROS Client Library (RCL)

The ROS Client Library layer provides common functionality across all client libraries, including:
- Node management
- Publisher/subscriber creation
- Service/client creation
- Parameter handling
- Logging

## Communication Patterns

### Publish-Subscribe Pattern

The publish-subscribe pattern is the most common communication pattern in ROS 2. Publishers send messages to topics, and subscribers receive messages from topics they're subscribed to.

### Client-Server Pattern

The client-server pattern is used for request/response communication. A client sends a request to a server, which processes the request and sends back a response.

### Action Pattern

The action pattern is used for long-running tasks that require feedback. An action client sends a goal to an action server, which executes the goal and provides feedback during execution.

## DDS Implementation Options

ROS 2 supports multiple DDS implementations, including:
- Fast DDS (formerly Fast RTPS)
- Cyclone DDS
- RTI Connext DDS
- Eclipse iceoryx (for intra-process communication)

Each implementation has its own strengths and is suitable for different use cases.

## Key Takeaways

- ROS 2 uses DDS middleware for communication between nodes
- QoS policies provide fine-grained control over communication behavior
- The architecture supports distributed computing across multiple machines
- Client libraries provide language-specific interfaces to the common ROS 2 functionality
- Multiple DDS implementations are supported for flexibility

## Key Concepts

- **DDS (Data Distribution Service)**: Middleware protocol for distributed, real-time data exchange
- **QoS Policies**: Configurable parameters that control communication behavior
- **Client Libraries**: Language-specific interfaces to ROS 2 functionality (rclpy, rclcpp)
- **Publish-Subscribe**: Communication pattern where publishers send messages to topics and subscribers receive them
- **Client-Server**: Synchronous request/response communication pattern
- **Action**: Asynchronous communication pattern with feedback and status updates

## Practical Exercises

1. Create a simple ROS 2 node that publishes messages to a topic
2. Create a subscriber that listens to the topic and logs received messages
3. Implement a service that performs a simple calculation
4. Test different QoS policies to observe their effects on communication

## Common Failure Modes

- **Network Configuration Issues**: DDS discovery may fail in complex network topologies
- **QoS Policy Mismatch**: Publishers and subscribers with incompatible QoS policies won't communicate
- **Resource Exhaustion**: Keeping too much history or using inappropriate QoS settings can consume excessive resources
- **Inconsistent Message Types**: Publishers and subscribers must use the same message types